"""
GARCH(1,1) Volatility Forecasting.

Models volatility clustering — volatile days tend to follow volatile days.
Produces forward-looking volatility forecasts for 5-day and 20-day horizons.
"""

import asyncio

import numpy as np
import structlog

logger = structlog.get_logger()


class GARCHForecaster:
    def __init__(self):
        pass

    async def forecast(
        self,
        returns: np.ndarray,
        horizons: list[int] | None = None,
    ) -> dict | None:
        """
        Fit GARCH(1,1) and forecast volatility.

        Args:
            returns: Log returns array (at least 100 observations).
            horizons: Forecast horizons in days. Default [5, 20].

        Returns:
            Dict with fitted parameters, current vol, forecasted vol per horizon.
        """
        return await asyncio.to_thread(self._forecast_sync, returns, horizons)

    def _forecast_sync(
        self,
        returns: np.ndarray,
        horizons: list[int] | None,
    ) -> dict | None:
        try:
            from arch import arch_model

            if horizons is None:
                horizons = [5, 20]

            returns = np.asarray(returns, dtype=float)
            returns = returns[np.isfinite(returns)]

            if len(returns) < 100:
                logger.warning("garch.insufficient_data", n=len(returns))
                return None

            # Convert to pandas Series for arch compatibility
            import pandas as pd
            returns_pct = pd.Series(returns * 100)

            # Fit GARCH(1,1)
            model = arch_model(
                returns_pct,
                vol="Garch",
                p=1,
                q=1,
                dist="t",  # Student-t for better tail risk capture
                rescale=False,
            )
            result = model.fit(disp="off", show_warning=False)

            # Extract parameters
            params = result.params
            omega = float(params.get("omega", 0))
            alpha = float(params.get("alpha[1]", 0))
            beta = float(params.get("beta[1]", 0))
            persistence = alpha + beta

            # Forecast path from arch itself: variance for days T+1 .. T+max_h,
            # conditional on everything up to and including the last return.
            # (The last *fitted* variance is the variance OF day T given T-1 —
            # it does not contain day T's shock, so forecasting from it lags
            # the market by one day and badly understates vol after a big move.)
            max_h = int(max(horizons))
            var_path = np.asarray(
                result.forecast(horizon=max_h, reindex=False).variance.iloc[-1].values,
                dtype=float,
            )
            if len(var_path) < max_h or not np.all(np.isfinite(var_path)) or np.any(var_path <= 0):
                raise ValueError("non-finite GARCH variance forecast")

            # Next-day conditional variance
            cond_var = float(var_path[0])

            # Long-run variance: omega / (1 - alpha - beta)
            if persistence < 0.999:
                long_run_var = omega / (1 - persistence)
            else:
                long_run_var = cond_var  # integrated GARCH: no finite long-run level

            # Historical realized volatility for comparison
            hist_vol_20d = float(np.std(returns[-20:]) * np.sqrt(252))
            hist_vol_60d = float(np.std(returns[-60:]) * np.sqrt(252))

            forecasts = {}
            for h in horizons:
                # Volatility over an h-day holding period = average of the
                # daily variance forecasts for days 1..h.
                forecast_var = float(np.mean(var_path[:h]))

                # Convert from pct^2 back to decimal
                forecast_vol_daily = np.sqrt(forecast_var) / 100
                forecast_vol_annual = forecast_vol_daily * np.sqrt(252)

                # Volatility ratio: predicted vs historical
                vol_ratio = forecast_vol_annual / hist_vol_60d if hist_vol_60d > 0 else 1.0

                forecasts[h] = {
                    "days": h,
                    "predicted_volatility_annual": round(float(forecast_vol_annual), 4),
                    "predicted_volatility_daily": round(float(forecast_vol_daily), 6),
                    "volatility_ratio": round(float(vol_ratio), 4),
                    "interpretation": self._interpret_vol_ratio(vol_ratio),
                }

            output = {
                "parameters": {
                    "omega": round(omega, 6),
                    "alpha": round(alpha, 4),
                    "beta": round(beta, 4),
                    "persistence": round(persistence, 4),
                },
                "current_conditional_vol_annual": round(
                    float(np.sqrt(cond_var) / 100 * np.sqrt(252)), 4
                ),
                "long_run_vol_annual": round(
                    float(np.sqrt(long_run_var) / 100 * np.sqrt(252)), 4
                ),
                "historical_vol_20d": round(hist_vol_20d, 4),
                "historical_vol_60d": round(hist_vol_60d, 4),
                "forecasts": forecasts,
                "n_observations": len(returns),
            }

            logger.info(
                "garch.complete",
                persistence=round(persistence, 4),
                current_vol=output["current_conditional_vol_annual"],
                forecast_5d=forecasts.get(5, {}).get("predicted_volatility_annual"),
            )
            return output

        except Exception as e:
            logger.warning("garch.fit_failed", error=str(e))
            # Fallback: EWMA (RiskMetrics) volatility estimate
            return self._ewma_fallback(returns, horizons)

    def _ewma_fallback(
        self,
        returns: np.ndarray,
        horizons: list[int] | None,
    ) -> dict | None:
        """EWMA (RiskMetrics λ=0.94) fallback when GARCH fails to converge."""
        if horizons is None:
            horizons = [5, 20]
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return None
        lam = 0.94
        var_t = float(np.var(returns[-20:]))
        # Recursive EWMA variance
        for r in returns:
            var_t = lam * var_t + (1 - lam) * r ** 2

        hist_vol_20d = float(np.std(returns[-20:]) * np.sqrt(252))
        hist_vol_60d = float(np.std(returns[-min(60, len(returns)):]) * np.sqrt(252))
        current_vol_annual = float(np.sqrt(var_t) * np.sqrt(252))

        forecasts = {}
        for h in horizons:
            # EWMA assumes constant forward vol
            forecast_vol_annual = current_vol_annual
            vol_ratio = forecast_vol_annual / hist_vol_60d if hist_vol_60d > 0 else 1.0
            forecasts[h] = {
                "days": h,
                "predicted_volatility_annual": round(float(forecast_vol_annual), 4),
                "predicted_volatility_daily": round(float(np.sqrt(var_t)), 6),
                "volatility_ratio": round(float(vol_ratio), 4),
                "interpretation": self._interpret_vol_ratio(vol_ratio),
            }

        logger.info("garch.ewma_fallback", current_vol=round(current_vol_annual, 4))
        return {
            "parameters": {
                "omega": 0.0, "alpha": round(1 - lam, 4),
                "beta": round(lam, 4), "persistence": 1.0,
            },
            "current_conditional_vol_annual": round(current_vol_annual, 4),
            "long_run_vol_annual": round(current_vol_annual, 4),
            "historical_vol_20d": round(hist_vol_20d, 4),
            "historical_vol_60d": round(hist_vol_60d, 4),
            "forecasts": forecasts,
            "n_observations": len(returns),
            "is_fallback": True,
        }

    def _interpret_vol_ratio(self, ratio: float) -> str:
        if ratio > 1.3:
            return "volatility_expanding"
        elif ratio < 0.7:
            return "volatility_contracting"
        return "volatility_stable"
