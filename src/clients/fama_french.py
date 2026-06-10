import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class FamaFrenchClient:
    """Loads Fama-French factor data from Kenneth French's data library."""

    FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def __init__(self):
        self._factors_cache: pd.DataFrame | None = None
        self._cache_date: datetime | None = None

    async def get_factors(self, days: int = 252) -> pd.DataFrame | None:
        """Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF, Mom."""
        try:
            # Cache for 24 hours
            if self._factors_cache is not None and self._cache_date:
                age = (datetime.now() - self._cache_date).total_seconds()
                if age < 86400:
                    return self._factors_cache.tail(days)

            df = await asyncio.to_thread(self._load_factors)
            if df is not None:
                self._factors_cache = df
                self._cache_date = datetime.now()
                return df.tail(days)
            return None
        except Exception as e:
            logger.warning("fama_french.load_failed", error=str(e))
            return None

    def _load_factors(self) -> pd.DataFrame | None:
        try:
            # Try pandas_datareader first
            try:
                import pandas_datareader.data as web
                ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")
                df = ff[0] / 100
                df.index = pd.to_datetime(df.index.astype(str))
                try:
                    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench")
                    mom_df = mom[0] / 100
                    df["Mom"] = mom_df.iloc[:, 0]
                except Exception:
                    df["Mom"] = 0.0
                return df
            except Exception:
                pass

            # Fallback: direct CSV download
            logger.info("fama_french.using_direct_download")
            df = pd.read_csv(
                self.FF_URL,
                skiprows=3,
                index_col=0,
                parse_dates=True,
            )
            # Clean column names and filter numeric rows
            df.columns = [c.strip() for c in df.columns]
            df.index.name = "Date"
            df = df[df.index.notna()]
            # Filter out non-numeric rows
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            df = df / 100  # Convert from percentages
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
            df = df[df.index.notna()]
            df["Mom"] = 0.0
            return df
        except Exception as e:
            logger.warning("fama_french.load_failed", error=str(e))
            return None

    @staticmethod
    def _to_naive_index(dates) -> pd.DatetimeIndex:
        """Convert a list of (possibly tz-aware) dates to a naive, normalized index."""
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx.normalize()

    def align_stock_to_factors(
        self,
        stock_returns: np.ndarray,
        stock_dates,
        factors: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex] | None:
        """
        Date-join stock returns to factor rows.

        FF factors are published with a multi-week lag, so tail-aligning the
        two arrays (factors.iloc[-n:] vs returns[-n:]) silently shifts every
        observation by that lag. This joins on actual dates instead.

        Args:
            stock_returns: Daily returns array.
            stock_dates: Dates aligned 1:1 with stock_returns.
            factors: Factor DataFrame indexed by date.

        Returns:
            (aligned_stock_returns, aligned_factor_rows, common_dates) or None.
        """
        if stock_dates is None or len(stock_dates) != len(stock_returns):
            return None
        idx = self._to_naive_index(stock_dates)
        stock = pd.Series(np.asarray(stock_returns, dtype=float), index=idx)
        # Drop any duplicate dates (defensive — splits/timezone edge cases)
        stock = stock[~stock.index.duplicated(keep="last")]
        common = factors.index.intersection(stock.index)
        if len(common) == 0:
            return None
        return stock.loc[common].values, factors.loc[common], common

    def compute_factor_exposure(
        self,
        stock_returns: np.ndarray,
        factors: pd.DataFrame,
        stock_dates=None,
    ) -> dict | None:
        """Run Fama-French regression. Returns alpha, betas, R-squared.

        Pass stock_dates (aligned 1:1 with stock_returns) to join on actual
        dates; without dates this falls back to tail alignment, which is
        biased whenever the FF data lags the price data.
        """
        try:
            aligned = self.align_stock_to_factors(stock_returns, stock_dates, factors)
            if aligned is not None:
                y, factor_rows, _ = aligned
                n = len(y)
                if n < 30:
                    return None
                X = factor_rows[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values
                rf = factor_rows["RF"].values
            else:
                logger.debug("fama_french.tail_alignment_fallback")
                n = min(len(stock_returns), len(factors))
                if n < 30:
                    return None
                y = stock_returns[-n:]
                X = factors.iloc[-n:][["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values
                rf = factors.iloc[-n:]["RF"].values
            y_excess = y - rf

            # Add intercept
            X_with_const = np.column_stack([np.ones(n), X])

            # OLS via normal equations
            beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y_excess, rcond=None)

            y_hat = X_with_const @ beta
            ss_res = np.sum((y_excess - y_hat) ** 2)
            ss_tot = np.sum((y_excess - np.mean(y_excess)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return {
                "alpha_daily": round(float(beta[0]), 6),
                "alpha_annual": round(float(beta[0] * 252), 4),
                "beta_market": round(float(beta[1]), 4),
                "beta_smb": round(float(beta[2]), 4),
                "beta_hml": round(float(beta[3]), 4),
                "beta_rmw": round(float(beta[4]), 4),
                "beta_cma": round(float(beta[5]), 4),
                "r_squared": round(float(r_squared), 4),
            }
        except Exception as e:
            logger.warning("fama_french.regression_failed", error=str(e))
            return None

    async def close(self):
        pass
