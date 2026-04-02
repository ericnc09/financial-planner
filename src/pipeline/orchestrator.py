import argparse
import asyncio
from datetime import datetime

import structlog
from sqlalchemy.exc import IntegrityError

from config.settings import Settings
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.monte_carlo import MonteCarloSimulator
from src.clients.congress import CongressClient
from src.clients.edgar import EdgarClient
from src.clients.fama_french import FamaFrenchClient
from src.clients.fred import FredClient
from src.clients.tiingo import TiingoClient
from src.clients.yahoo import YahooClient
from src.models.database import (
    ConvictionScore,
    Direction,
    Enrichment,
    EventStudyResult,
    ExtendedMacroData,
    FamaFrenchExposure,
    GARCHForecast,
    HMMRegimeState,
    MacroSnapshot,
    MonteCarloResult,
    SmartMoneyEvent,
    SourceType,
    get_session_factory,
    init_db,
)
from src.models.schemas import MacroRegime
from src.scoring.conviction_engine import ConvictionEngine
from src.scoring.fundamental_scorer import FundamentalScorer
from src.scoring.macro_scorer import MacroScorer
from src.scoring.signal_scorer import SignalScorer

logger = structlog.get_logger()


class Orchestrator:
    """Pipeline: ingest → enrich → macro → score. No execution — manual trades."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.edgar: EdgarClient | None = None
        self.congress: CongressClient | None = None
        self.tiingo: TiingoClient | None = None
        self.fred: FredClient | None = None
        self.yahoo: YahooClient | None = None
        self.fama_french: FamaFrenchClient | None = None
        self.monte_carlo: MonteCarloSimulator | None = None
        self.hmm: HMMRegimeDetector | None = None
        self.garch: GARCHForecaster | None = None
        self.event_study: EventStudyAnalyzer | None = None
        self.engine = None
        self.session_factory = None
        self.conviction_engine: ConvictionEngine | None = None

    async def initialize(self):
        logger.info("orchestrator.initializing")

        # Database
        self.engine = init_db(self.settings.database_url)
        self.session_factory = get_session_factory(self.engine)

        # API clients (edgar + congress are free, no keys needed)
        self.edgar = EdgarClient()
        self.congress = CongressClient()
        self.tiingo = TiingoClient(self.settings.tiingo_api_key)
        self.fred = FredClient(self.settings.fred_api_key)
        self.yahoo = YahooClient()
        self.fama_french = FamaFrenchClient()

        # Analysis models
        self.monte_carlo = MonteCarloSimulator(n_simulations=10_000)
        self.hmm = HMMRegimeDetector()
        self.garch = GARCHForecaster()
        self.event_study = EventStudyAnalyzer()

        # Scoring
        signal_scorer = SignalScorer()
        fundamental_scorer = FundamentalScorer()
        macro_scorer = MacroScorer()
        self.conviction_engine = ConvictionEngine(
            self.settings, signal_scorer, fundamental_scorer, macro_scorer
        )

        logger.info("orchestrator.initialized")

    async def run_cycle(self):
        logger.info("orchestrator.cycle_start")

        # --- Step 1: INGEST (SEC EDGAR + House/Senate Stock Watcher) ---
        insider_signals = await self.edgar.get_bulk_insider_trades(since_days=14)
        congressional_signals = await self.congress.get_all_congressional_trades(
            since_days=14
        )
        signals = insider_signals + congressional_signals
        signals.sort(key=lambda e: e.trade_date, reverse=True)
        new_events = self._persist_events(signals)
        logger.info(
            "orchestrator.ingested", new=len(new_events), total=len(signals)
        )

        if not new_events:
            logger.info("orchestrator.no_new_events")
            return

        # --- Step 2: ENRICH ---
        enrichments = await self._enrich_events(new_events)
        logger.info("orchestrator.enriched", count=len(enrichments))

        # --- Step 3: MACRO ---
        macro_snapshot = await asyncio.to_thread(self.fred.get_macro_snapshot)
        self._persist_macro_snapshot(macro_snapshot)

        # --- Step 3b: EXTENDED MACRO ---
        try:
            extended_macro = await asyncio.to_thread(self.fred.get_extended_macro)
            self._persist_extended_macro(extended_macro)
            logger.info("orchestrator.extended_macro_saved", data=extended_macro)
        except Exception as e:
            logger.warning("orchestrator.extended_macro_failed", error=str(e))

        # --- Step 3c: ANALYSIS MODELS (Monte Carlo, HMM, GARCH, Fama-French) ---
        unique_tickers = list({schema.ticker for _, schema in new_events})
        await self._run_analysis_models(unique_tickers)

        # --- Step 3d: EVENT STUDY ---
        await self._run_event_studies(new_events)

        # --- Step 4: SCORE ---
        scored = []
        for event_id, event_schema, enrichment_schema in enrichments:
            result = self.conviction_engine.compute(
                event_schema, enrichment_schema, macro_snapshot, signals
            )
            self._persist_conviction_score(event_id, result)
            scored.append((event_schema, enrichment_schema, result))

        passing = [s for s in scored if s[2].passes_threshold]
        logger.info(
            "orchestrator.scored",
            total=len(scored),
            passing=len(passing),
            threshold=self.settings.conviction_threshold,
        )

        # Log actionable signals for manual review
        for event, enrichment, result in passing:
            logger.info(
                "orchestrator.actionable_signal",
                ticker=event.ticker,
                actor=event.actor,
                direction=event.direction.value,
                conviction=result.conviction,
                price=enrichment.price_at_signal,
                sector=enrichment.sector,
            )

        logger.info("orchestrator.cycle_complete")

    def _persist_events(self, signals):
        session = self.session_factory()
        new_events = []
        try:
            for sig in signals:
                event = SmartMoneyEvent(
                    ticker=sig.ticker,
                    actor=sig.actor,
                    direction=Direction(sig.direction.value),
                    size_lower=sig.size_lower,
                    size_upper=sig.size_upper,
                    trade_date=sig.trade_date,
                    disclosure_date=sig.disclosure_date,
                    source_type=SourceType(sig.source_type.value),
                    raw_payload=sig.raw_payload,
                )
                session.add(event)
                try:
                    session.flush()
                    new_events.append((event.id, sig))
                except IntegrityError:
                    session.rollback()
            session.commit()
        finally:
            session.close()
        return new_events

    async def _enrich_events(self, new_events):
        results = []
        # Deduplicate: enrich each ticker once, reuse for all events with same ticker
        ticker_cache: dict[str, object] = {}
        unique_tickers = list({schema.ticker for _, schema in new_events})
        logger.info(
            "orchestrator.enriching",
            unique_tickers=len(unique_tickers),
            total_events=len(new_events),
        )

        for ticker in unique_tickers:
            try:
                enrichment = await self.tiingo.enrich_ticker(ticker)
                ticker_cache[ticker] = enrichment
                logger.info("orchestrator.enriched_ticker", ticker=ticker)
            except Exception as e:
                logger.warning(
                    "orchestrator.enrich_failed",
                    ticker=ticker,
                    error=str(e),
                )

        for event_id, event_schema in new_events:
            enrichment = ticker_cache.get(event_schema.ticker)
            if enrichment:
                self._persist_enrichment(event_id, enrichment)
                results.append((event_id, event_schema, enrichment))

        return results

    def _persist_enrichment(self, event_id, enrichment_schema):
        session = self.session_factory()
        try:
            enr = Enrichment(
                event_id=event_id,
                pe_ratio=enrichment_schema.pe_ratio,
                market_cap=enrichment_schema.market_cap,
                revenue_growth_yoy=enrichment_schema.revenue_growth_yoy,
                eps_latest=enrichment_schema.eps_latest,
                eps_growth_yoy=enrichment_schema.eps_growth_yoy,
                price_at_signal=enrichment_schema.price_at_signal,
                momentum_30d=enrichment_schema.momentum_30d,
                momentum_90d=enrichment_schema.momentum_90d,
                rsi_14d=enrichment_schema.rsi_14d,
                drawdown_from_52w_high=enrichment_schema.drawdown_from_52w_high,
                avg_volume_30d=enrichment_schema.avg_volume_30d,
                sector=enrichment_schema.sector,
            )
            session.add(enr)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_macro_snapshot(self, snapshot):
        session = self.session_factory()
        try:
            ms = MacroSnapshot(
                snapshot_date=snapshot.snapshot_date or datetime.utcnow(),
                yield_spread_10y2y=snapshot.yield_spread_10y2y,
                unemployment_claims=snapshot.unemployment_claims,
                cpi_yoy=snapshot.cpi_yoy,
                fed_funds_rate=snapshot.fed_funds_rate,
                regime=MacroRegime(snapshot.regime.value)
                if snapshot.regime
                else None,
                regime_score=snapshot.regime_score,
            )
            session.add(ms)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_conviction_score(self, event_id, result):
        session = self.session_factory()
        try:
            cs = ConvictionScore(
                event_id=event_id,
                signal_score=result.signal_score,
                fundamental_score=result.fundamental_score,
                macro_modifier=result.macro_modifier,
                direction_boost=result.direction_boost,
                conviction=result.conviction,
                passes_threshold=result.passes_threshold,
            )
            session.add(cs)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    async def _run_analysis_models(self, tickers: list[str]):
        """Run Monte Carlo, HMM, GARCH, and Fama-French on each unique ticker."""
        logger.info("orchestrator.analysis_start", tickers=len(tickers))
        run_date = datetime.utcnow()

        # Pre-load Fama-French factors once
        ff_factors = await self.fama_french.get_factors(days=252)

        for ticker in tickers:
            try:
                # Get price data from yfinance (free, no rate limit)
                price_data = await self.yahoo.get_price_history(ticker, days=504)
                if price_data is None:
                    logger.warning("orchestrator.analysis_no_data", ticker=ticker)
                    continue

                closes = price_data["closes"]
                returns = price_data["returns"]
                volumes = price_data["volumes"]

                # Monte Carlo
                try:
                    mc_result = self.monte_carlo.simulate(closes, horizons=[21, 63])
                    if mc_result:
                        self._persist_monte_carlo(ticker, run_date, mc_result)
                except Exception as e:
                    logger.warning("orchestrator.mc_failed", ticker=ticker, error=str(e))

                # HMM Regime Detection
                try:
                    hmm_result = await self.hmm.fit_and_predict(returns, volumes)
                    if hmm_result:
                        self._persist_hmm(ticker, run_date, hmm_result)
                except Exception as e:
                    logger.warning("orchestrator.hmm_failed", ticker=ticker, error=str(e))

                # GARCH Volatility Forecast
                try:
                    garch_result = await self.garch.forecast(returns, horizons=[5, 20])
                    if garch_result:
                        self._persist_garch(ticker, run_date, garch_result)
                except Exception as e:
                    logger.warning("orchestrator.garch_failed", ticker=ticker, error=str(e))

                # Fama-French Factor Exposure
                try:
                    if ff_factors is not None:
                        ff_result = self.fama_french.compute_factor_exposure(returns, ff_factors)
                        if ff_result:
                            self._persist_fama_french(ticker, run_date, ff_result)
                except Exception as e:
                    logger.warning("orchestrator.ff_failed", ticker=ticker, error=str(e))

                logger.info("orchestrator.analysis_complete", ticker=ticker)
            except Exception as e:
                logger.warning("orchestrator.analysis_ticker_failed", ticker=ticker, error=str(e))

        logger.info("orchestrator.analysis_done", tickers=len(tickers))

    def _persist_monte_carlo(self, ticker: str, run_date: datetime, result: dict):
        session = self.session_factory()
        try:
            h30 = result["horizons"].get(21, {})
            h90 = result["horizons"].get(63, {})
            mc = MonteCarloResult(
                ticker=ticker,
                run_date=run_date,
                current_price=result["current_price"],
                annual_drift=result["annual_drift"],
                annual_volatility=result["annual_volatility"],
                n_simulations=result["n_simulations"],
                h30_p10=h30.get("percentiles", {}).get("p10"),
                h30_p25=h30.get("percentiles", {}).get("p25"),
                h30_p50=h30.get("percentiles", {}).get("p50"),
                h30_p75=h30.get("percentiles", {}).get("p75"),
                h30_p90=h30.get("percentiles", {}).get("p90"),
                h30_prob_profit=h30.get("probability_of_profit"),
                h30_expected_return=h30.get("expected_return"),
                h30_var_95=h30.get("value_at_risk_95"),
                h90_p10=h90.get("percentiles", {}).get("p10"),
                h90_p25=h90.get("percentiles", {}).get("p25"),
                h90_p50=h90.get("percentiles", {}).get("p50"),
                h90_p75=h90.get("percentiles", {}).get("p75"),
                h90_p90=h90.get("percentiles", {}).get("p90"),
                h90_prob_profit=h90.get("probability_of_profit"),
                h90_expected_return=h90.get("expected_return"),
                h90_var_95=h90.get("value_at_risk_95"),
            )
            session.add(mc)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_hmm(self, ticker: str, run_date: datetime, result: dict):
        session = self.session_factory()
        try:
            current = result["current_state"]
            probs = result["current_probabilities"]
            trans = result["transition_matrix"].get(current, {})
            hmm = HMMRegimeState(
                ticker=ticker,
                run_date=run_date,
                current_state=current,
                prob_bull=probs.get("bull"),
                prob_bear=probs.get("bear"),
                prob_sideways=probs.get("sideways"),
                trans_to_bull=trans.get("bull"),
                trans_to_bear=trans.get("bear"),
                trans_to_sideways=trans.get("sideways"),
                n_observations=result["n_observations"],
            )
            session.add(hmm)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_garch(self, ticker: str, run_date: datetime, result: dict):
        session = self.session_factory()
        try:
            f5 = result["forecasts"].get(5, {})
            f20 = result["forecasts"].get(20, {})
            gf = GARCHForecast(
                ticker=ticker,
                run_date=run_date,
                omega=result["parameters"]["omega"],
                alpha=result["parameters"]["alpha"],
                beta=result["parameters"]["beta"],
                persistence=result["parameters"]["persistence"],
                current_vol_annual=result["current_conditional_vol_annual"],
                long_run_vol_annual=result["long_run_vol_annual"],
                historical_vol_20d=result["historical_vol_20d"],
                historical_vol_60d=result["historical_vol_60d"],
                forecast_5d_vol=f5.get("predicted_volatility_annual"),
                forecast_5d_ratio=f5.get("volatility_ratio"),
                forecast_5d_interpretation=f5.get("interpretation"),
                forecast_20d_vol=f20.get("predicted_volatility_annual"),
                forecast_20d_ratio=f20.get("volatility_ratio"),
                forecast_20d_interpretation=f20.get("interpretation"),
                n_observations=result["n_observations"],
            )
            session.add(gf)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_fama_french(self, ticker: str, run_date: datetime, result: dict):
        session = self.session_factory()
        try:
            ff = FamaFrenchExposure(
                ticker=ticker,
                run_date=run_date,
                alpha_daily=result["alpha_daily"],
                alpha_annual=result["alpha_annual"],
                beta_market=result["beta_market"],
                beta_smb=result["beta_smb"],
                beta_hml=result["beta_hml"],
                beta_rmw=result["beta_rmw"],
                beta_cma=result["beta_cma"],
                r_squared=result["r_squared"],
            )
            session.add(ff)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    async def _run_event_studies(self, new_events: list):
        """Run event study analysis on each new event."""
        import json
        import numpy as np

        logger.info("orchestrator.event_study_start", events=len(new_events))
        run_date = datetime.utcnow()

        # Pre-load Fama-French factors for market returns
        ff_factors = await self.fama_french.get_factors(days=504)
        if ff_factors is None:
            logger.warning("orchestrator.event_study_no_factors")
            return

        market_returns = ff_factors["Mkt-RF"].values

        # Cache price data per ticker
        ticker_price_cache: dict[str, dict] = {}
        results = []

        for event_id, event_schema in new_events:
            try:
                ticker = event_schema.ticker

                # Fetch price data (cached per ticker)
                if ticker not in ticker_price_cache:
                    price_data = await self.yahoo.get_price_history(ticker, days=504)
                    ticker_price_cache[ticker] = price_data

                price_data = ticker_price_cache[ticker]
                if price_data is None:
                    continue

                price_dates = price_data["dates"]
                price_returns = price_data["returns"]

                # Align market returns to the same length
                n = min(len(price_returns), len(market_returns))
                aligned_market = market_returns[-n:]
                aligned_stock = price_returns[-n:]
                aligned_dates = price_dates[-(n + 1):]  # dates are 1 longer than returns

                result = self.event_study.analyze_event(
                    ticker=ticker,
                    event_date=event_schema.trade_date,
                    direction=event_schema.direction.value,
                    price_dates=aligned_dates,
                    price_returns=aligned_stock,
                    market_returns=aligned_market,
                )

                if result:
                    result["event_id"] = event_id
                    result["source_type"] = event_schema.source_type.value
                    self._persist_event_study(result, run_date)
                    results.append(result)

            except Exception as e:
                logger.warning(
                    "orchestrator.event_study_failed",
                    ticker=event_schema.ticker,
                    error=str(e),
                )

        if results:
            aggregate = self.event_study.aggregate_results(results)
            logger.info(
                "orchestrator.event_study_done",
                n_analyzed=len(results),
                mean_car_5d=aggregate.get("overall", {}).get("car_5d", {}).get("mean"),
                mean_car_20d=aggregate.get("overall", {}).get("car_20d", {}).get("mean"),
            )
        else:
            logger.info("orchestrator.event_study_no_results")

    def _persist_event_study(self, result: dict, run_date: datetime):
        import json

        session = self.session_factory()
        try:
            es = EventStudyResult(
                ticker=result["ticker"],
                event_id=result["event_id"],
                run_date=run_date,
                direction=result["direction"],
                source_type=result.get("source_type"),
                car_1d=result["car_1d"],
                car_5d=result["car_5d"],
                car_10d=result["car_10d"],
                car_20d=result["car_20d"],
                t_statistic=result["t_statistic"],
                p_value=result["p_value"],
                is_significant=result["is_significant"],
                alpha=result["alpha"],
                beta=result["beta"],
                sigma_est=result["sigma_est"],
                n_estimation=result["n_estimation"],
                daily_cars=json.dumps(result["daily_cars"]),
            )
            session.add(es)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_extended_macro(self, data: dict):
        session = self.session_factory()
        try:
            em = ExtendedMacroData(
                snapshot_date=datetime.utcnow(),
                vix=data.get("vix"),
                consumer_sentiment=data.get("consumer_sentiment"),
                money_supply_m2=data.get("money_supply_m2"),
                housing_starts=data.get("housing_starts"),
                industrial_production=data.get("industrial_production"),
            )
            session.add(em)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    async def run_oneshot(self):
        await self.initialize()
        await self.run_cycle()
        await self.shutdown()

    async def run_daemon(self):
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        await self.initialize()
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self.run_cycle,
            "interval",
            minutes=self.settings.schedule_interval_minutes,
        )
        scheduler.start()
        logger.info(
            "orchestrator.daemon_started",
            interval_min=self.settings.schedule_interval_minutes,
        )
        await self.run_cycle()
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            await self.shutdown()

    async def shutdown(self):
        logger.info("orchestrator.shutting_down")
        if self.edgar:
            await self.edgar.close()
        if self.congress:
            await self.congress.close()
        if self.tiingo:
            await self.tiingo.close()
        if self.yahoo:
            await self.yahoo.close()
        if self.fama_french:
            await self.fama_french.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Money Follows Pipeline")
    parser.add_argument(
        "--mode",
        choices=["oneshot", "daemon"],
        default=None,
        help="Run mode (overrides settings)",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.mode:
        settings.pipeline_mode = args.mode

    orchestrator = Orchestrator(settings)

    if settings.pipeline_mode == "daemon":
        asyncio.run(orchestrator.run_daemon())
    else:
        asyncio.run(orchestrator.run_oneshot())
