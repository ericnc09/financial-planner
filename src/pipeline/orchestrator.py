import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from config.settings import Settings
from src.analysis.bayesian_decay import BayesianDecayAnalyzer, decay_component
from src.analysis.copula_tail_risk import CopulaTailRisk
from src.analysis.earnings_overlay import EarningsOverlay
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.mean_variance import MeanVarianceOptimizer
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.news_sentiment import NewsSentimentAnalyzer
from src.analysis.pooled_alpha import (
    pooled_event_stats,
    post_event_abnormal_returns,
    trading_days_since,
)
from src.analysis.weather_overlay import BASELINE_WINDOW_DAYS, WeatherOverlay
from src.clients.congress import CongressClient
from src.clients.options import METHOD_VERSION as OPTIONS_METHOD_VERSION
from src.clients.options import OptionsClient, percentile_rank
from src.clients.edgar import EdgarClient
from src.clients.edgar_13f import Edgar13FClient
from src.clients.edgar_xbrl import EdgarXbrlClient
from src.clients.fama_french import FamaFrenchClient
from src.clients.fred import FredClient
from src.clients.macro_extra import MacroExtraClient
from src.clients.news import NewsClient
from src.clients.tiingo import TiingoClient
from src.clients.yahoo import YahooClient
from src.models.database import (
    ConvictionScore,
    Direction,
    Enrichment,
    BayesianDecayResult,
    CopulaTailRiskResult,
    EnsembleScoreResult,
    EventStudyResult,
    ExtendedMacroData,
    MacroExtraData,
    MeanVarianceResult,
    NewsSentimentResult,
    OptionsFlow,
    FamaFrenchExposure,
    GARCHForecast,
    HMMRegimeState,
    MacroSnapshot,
    MonteCarloResult,
    SmartMoneyEvent,
    SourceType,
    WeatherSnapshot,
    get_session_factory,
    init_db,
)
from src.models.schemas import MacroRegime
from src.alerts.digest import publish_digest
from src.alerts.notifier import SlackNotifier, WhatsAppNotifier, create_notifier
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
        self.copula: CopulaTailRisk | None = None
        self.bayesian_decay: BayesianDecayAnalyzer | None = None
        self.mean_variance: MeanVarianceOptimizer | None = None
        self.ensemble: EnsembleScorer | None = None
        self.options_client: OptionsClient | None = None
        self.earnings_overlay: EarningsOverlay | None = None
        self.news_client: NewsClient | None = None
        self.news_analyzer: NewsSentimentAnalyzer | None = None
        self.weather: WeatherOverlay | None = None
        self.edgar_xbrl: EdgarXbrlClient | None = None
        self.edgar_13f: Edgar13FClient | None = None
        self.macro_extra: MacroExtraClient | None = None
        self.notifier: SlackNotifier | WhatsAppNotifier | None = None
        self.engine = None
        self.session_factory = None
        self.conviction_engine: ConvictionEngine | None = None
        # ticker -> "common" | "preferred_or_debt" | "fund_or_other", filled
        # during enrichment. Equity factor models are skipped for non-common.
        self._security_types: dict[str, str | None] = {}

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
        self.copula = CopulaTailRisk()
        self.bayesian_decay = BayesianDecayAnalyzer()
        self.mean_variance = MeanVarianceOptimizer()
        self.ensemble = EnsembleScorer()
        self.options_client = OptionsClient()
        self.earnings_overlay = EarningsOverlay()
        self.news_client = NewsClient(getattr(self.settings, "finnhub_api_key", None))
        self.news_analyzer = NewsSentimentAnalyzer()
        self.weather = WeatherOverlay()
        self.edgar_xbrl = EdgarXbrlClient()
        self.edgar_13f = Edgar13FClient()
        self.macro_extra = MacroExtraClient(getattr(self.settings, "bls_api_key", None))

        # Alerts
        self.notifier = create_notifier(self.settings)

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

        # --- Step 1: INGEST (SEC EDGAR + House/Senate Stock Watcher + 13F) ---
        insider_signals = await self.edgar.get_bulk_insider_trades(since_days=14)
        congressional_signals = await self.congress.get_all_congressional_trades(
            since_days=14
        )
        # 13F institutional holdings diffs (quarterly cadence; the events
        # dedup against the unique constraint on re-runs)
        institutional_signals = []
        try:
            institutional_signals = await self.edgar_13f.get_recent_13f_events()
        except Exception as e:
            logger.warning("orchestrator.13f_failed", error=str(e))
        signals = insider_signals + congressional_signals + institutional_signals
        signals.sort(key=lambda e: e.trade_date, reverse=True)
        new_events = self._persist_events(signals)
        logger.info(
            "orchestrator.ingested", new=len(new_events), total=len(signals)
        )

        if not new_events:
            logger.info("orchestrator.no_new_events")
            # Returns keep accruing even when no new signals arrive
            await self._update_signal_performance()
            # ...and so do completed post-disclosure windows
            await self._run_event_studies()
            # Still produce the morning top pick from recent signals in the DB
            await self._log_top_pick()
            await self._publish_digest()
            return

        # --- Step 2: ENRICH ---
        enrichments = await self._enrich_events(new_events)
        logger.info("orchestrator.enriched", count=len(enrichments))

        # Publish now: the digest only needs events and sectors, and the models
        # below can take over an hour, past the 7:30 AM email.
        await self._publish_digest()

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

        # --- Step 3b2: BLS + TREASURY MACRO (enriches the FRED layer) ---
        try:
            macro_extra = await self.macro_extra.get_snapshot()
            self._persist_macro_extra(macro_extra)
        except Exception as e:
            logger.warning("orchestrator.macro_extra_failed", error=str(e))

        # --- Step 3c: ANALYSIS MODELS (Monte Carlo, HMM, GARCH, Fama-French) ---
        unique_tickers = list({schema.ticker for _, schema in new_events})
        await self._run_analysis_models(unique_tickers)

        # --- Step 3d: COPULA TAIL RISK ---
        await self._run_copula_analysis(unique_tickers)

        # --- Step 3f: EVENT STUDIES for signals whose window has completed ---
        # (feeds the pooled event-study and decay components scored in 3g)
        await self._run_event_studies()

        # --- Step 3h: OPTIONS FLOW ---
        await self._run_options_analysis(unique_tickers)

        # --- Step 3g: ENSEMBLE SCORING ---
        await self._run_ensemble_scoring(new_events)

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

        # Log actionable signals
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

        # --- Step 4b: MEAN-VARIANCE OPTIMIZATION ---
        # Runs after scoring so the long-only portfolio is built from the
        # highest-conviction BUY signals, not from every ticker in the cycle
        # (sells and hundreds of names made the problem meaningless and
        # numerically unstable).
        try:
            await self._run_mean_variance(self._portfolio_candidates(scored))
        except Exception as e:
            logger.warning("orchestrator.meanvar_failed", error=str(e))

        # --- Step 5: PERFORMANCE TRACKING ---
        await self._update_signal_performance()

        # --- Step 6: TOP PICK ---
        await self._log_top_pick()

        logger.info("orchestrator.cycle_complete")

    async def _update_signal_performance(self):
        """Refresh realized 5/10/20/60d returns for past signals. Without this
        the live track record silently goes stale between manual runs."""
        try:
            from src.tracking.performance import PerformanceTracker

            tracker = PerformanceTracker(self.session_factory)
            updated = await tracker.update_performance()
            logger.info("orchestrator.performance_updated", rows=updated)
        except Exception as e:
            logger.warning("orchestrator.performance_update_failed", error=str(e))

    async def _publish_digest(self):
        """Publish today's trades, and send the one compiled daily message."""
        try:
            await publish_digest(self.session_factory, self.notifier, self.settings)
        except Exception as e:
            logger.warning("orchestrator.digest_failed", error=str(e))

    async def _log_top_pick(self):
        """Select and log the single most promising stock for the following
        days, blending conviction, ensemble, and Monte Carlo."""
        try:
            from src.analysis.top_pick import select_top_pick

            pick = select_top_pick(self.session_factory, lookback_days=7)
            if pick is None:
                logger.info("orchestrator.top_pick_none")
                return
            logger.info(
                "orchestrator.top_pick",
                ticker=pick["ticker"], score=pick["score"],
                sector=pick.get("sector"),
            )
        except Exception as e:
            logger.warning("orchestrator.top_pick_failed", error=str(e))

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
                try:
                    # Roll back only this insert when the disclosure is already
                    # present. A full session rollback also discards earlier
                    # new events while leaving their IDs in new_events.
                    with session.begin_nested():
                        session.add(event)
                        session.flush()
                except IntegrityError:
                    continue
                new_events.append((event.id, sig))
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
                # Use yfinance (free, no rate limits) instead of Tiingo
                enrichment = await self.yahoo.enrich_ticker(ticker)
                if enrichment is None:
                    # Fallback to Tiingo if yfinance fails
                    enrichment = await self.tiingo.enrich_ticker(ticker)
                # Overlay SEC XBRL fundamentals — point-in-time correct and
                # far more reliable than yfinance .info scraping
                if enrichment is not None:
                    enrichment = await self._overlay_xbrl_fundamentals(ticker, enrichment)
                ticker_cache[ticker] = enrichment
                if enrichment is not None:
                    self._security_types[ticker] = getattr(enrichment, "security_type", None)
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

    async def _overlay_xbrl_fundamentals(self, ticker: str, enrichment):
        """Replace yfinance fundamental fields with SEC XBRL values when
        available. Price-derived fields (momentum, RSI, etc.) stay yfinance."""
        try:
            xbrl = await self.edgar_xbrl.get_fundamentals(ticker)
            if not xbrl:
                return enrichment
            update = {}
            if xbrl.get("eps_latest") is not None:
                update["eps_latest"] = xbrl["eps_latest"]
            if xbrl.get("eps_growth_yoy") is not None:
                update["eps_growth_yoy"] = xbrl["eps_growth_yoy"]
            if xbrl.get("revenue_growth_yoy") is not None:
                update["revenue_growth_yoy"] = xbrl["revenue_growth_yoy"]
            # Recompute trailing P/E from price and XBRL TTM EPS
            eps_ttm = xbrl.get("eps_ttm")
            if eps_ttm and eps_ttm > 0 and enrichment.price_at_signal:
                update["pe_ratio"] = round(enrichment.price_at_signal / eps_ttm, 2)
            if update:
                enrichment = enrichment.model_copy(update=update)
                logger.info("orchestrator.xbrl_overlay", ticker=ticker,
                            fields=sorted(update.keys()))
            return enrichment
        except Exception as e:
            logger.debug("orchestrator.xbrl_overlay_failed", ticker=ticker, error=str(e))
            return enrichment

    def _persist_macro_extra(self, snapshot: dict):
        import json
        session = self.session_factory()
        try:
            row = MacroExtraData(
                snapshot_date=datetime.utcnow(),
                payrolls_yoy_pct=snapshot.get("payrolls_yoy_pct"),
                unemployment_rate=snapshot.get("unemployment_rate"),
                treasury_avg_rate=snapshot.get("treasury_avg_rate"),
                total_public_debt=snapshot.get("total_public_debt"),
                detail_json=json.dumps(snapshot.get("detail") or {}),
            )
            session.add(row)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_news_sentiment(self, ticker: str, run_date: datetime, result: dict):
        import json
        session = self.session_factory()
        try:
            row = NewsSentimentResult(
                ticker=ticker,
                run_date=run_date,
                as_of=datetime.fromisoformat(result["as_of"]),
                n_articles=result.get("n_articles"),
                n_articles_3d=result.get("n_articles_3d"),
                news_volume_z=result.get("news_volume_z"),
                news_sentiment_mean=result.get("news_sentiment_mean"),
                news_sentiment_trend_3d=result.get("news_sentiment_trend_3d"),
                sentiment_model=result.get("sentiment_model"),
                earliest_article_at=datetime.fromisoformat(result["earliest_article_at"])
                if result.get("earliest_article_at") else None,
                latest_article_at=datetime.fromisoformat(result["latest_article_at"])
                if result.get("latest_article_at") else None,
                articles_json=json.dumps(result.get("articles") or []),
            )
            session.add(row)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

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
                short_ratio=enrichment_schema.short_ratio,
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

                # Fama-French Factor Exposure (date-joined to FF factor rows).
                # Equity factors say nothing about a preferred share or a note.
                try:
                    if ff_factors is not None and self._is_common_stock(ticker):
                        return_dates = price_data["dates"][1:] if price_data.get("dates") else None
                        ff_result = self.fama_french.compute_factor_exposure(
                            returns, ff_factors, stock_dates=return_dates
                        )
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
            # Folded bull/bear/sideways figures: the raw per-state dict has
            # extra keys (mild_bull, mild_bear) when BIC picks 4-5 states.
            current = result["current_state"]
            trans = result.get("transition_from_current") or {}
            hmm = HMMRegimeState(
                ticker=ticker,
                run_date=run_date,
                current_state=current,
                prob_bull=result.get("prob_bull"),
                prob_bear=result.get("prob_bear"),
                prob_sideways=result.get("prob_sideways"),
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
                alpha_tstat=result.get("alpha_tstat"),
            )
            session.add(ff)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    # --- Event studies: measured from disclosure, on completed windows ---
    EVENT_STUDY_MIN_AGE_DAYS = 32     # ~21 trading days must have passed
    EVENT_STUDY_MAX_AGE_DAYS = 300    # older events fall outside the price window
    EVENT_STUDY_GIVE_UP_DAYS = 60     # still unmeasurable by then: mark, stop retrying
    EVENT_STUDY_MAX_TICKERS = 500     # per cycle (~1 min); the backlog drains in a few runs
    EVENT_STUDY_HISTORY_DAYS = 730
    MARKET_PROXY = "SPY"

    async def _run_event_studies(self):
        """Maturation job: event studies for past signals whose 20-day
        post-disclosure window has completed.

        Day 0 is the first trading day AFTER the signal became public — the
        first day a follower could trade — and the market model uses SPY, so
        nothing waits on the weeks-late Fama-French file. A signal's own event
        study can therefore never exist while the signal is fresh; fresh
        signals are scored from the pooled history instead (see
        _pooled_signal_alpha).
        """
        try:
            now = datetime.utcnow()
            session = self.session_factory()
            try:
                signal_date = func.coalesce(
                    SmartMoneyEvent.disclosure_date, SmartMoneyEvent.trade_date
                )
                done = session.query(EventStudyResult.event_id).filter(
                    EventStudyResult.anchor == "disclosure"
                )
                events = (
                    session.query(SmartMoneyEvent)
                    .filter(signal_date <= now - timedelta(days=self.EVENT_STUDY_MIN_AGE_DAYS))
                    .filter(signal_date >= now - timedelta(days=self.EVENT_STUDY_MAX_AGE_DAYS))
                    .filter(~SmartMoneyEvent.id.in_(done))
                    .order_by(signal_date.desc())
                    .all()
                )
                by_ticker: dict[str, list[tuple]] = {}
                for e in events:
                    by_ticker.setdefault(e.ticker, []).append((
                        e.id, e.direction.value, e.source_type.value,
                        e.disclosure_date or e.trade_date,
                    ))
            finally:
                session.close()

            if not by_ticker:
                logger.info("orchestrator.event_study_nothing_pending")
                return

            tickers = list(by_ticker)[: self.EVENT_STUDY_MAX_TICKERS]
            logger.info(
                "orchestrator.event_study_start",
                tickers=len(tickers), backlog_tickers=len(by_ticker),
            )

            market = await self.yahoo.get_price_history(
                self.MARKET_PROXY, days=self.EVENT_STUDY_HISTORY_DAYS
            )
            if market is None or not market.get("dates"):
                logger.warning("orchestrator.event_study_no_market_data")
                return
            market_series = self._return_series(market).rename("market")

            measured = 0
            for ticker in tickers:
                try:
                    price_data = await self.yahoo.get_price_history(
                        ticker, days=self.EVENT_STUDY_HISTORY_DAYS
                    )
                    aligned = None
                    if price_data is not None and price_data.get("dates"):
                        # Date-join stock and market returns
                        frame = pd.concat(
                            [self._return_series(price_data).rename("stock"), market_series],
                            axis=1, join="inner",
                        ).dropna()
                        if len(frame) > 0:
                            aligned = frame

                    for event_id, direction, source_type, disclosed in by_ticker[ticker]:
                        result = None
                        if aligned is not None:
                            common = list(aligned.index)
                            result = self.event_study.analyze_event(
                                ticker=ticker,
                                # first trading day strictly after disclosure
                                event_date=disclosed + timedelta(days=1),
                                direction=direction,
                                # analyze_event maps return i to price_dates[i+1]
                                price_dates=[common[0]] + common,
                                price_returns=aligned["stock"].values,
                                market_returns=aligned["market"].values,
                            )
                        if result:
                            result.update(
                                event_id=event_id, source_type=source_type,
                                anchor="disclosure", anchor_date=disclosed,
                            )
                            self._persist_event_study(result, now)
                            measured += 1
                        elif (now - disclosed).days > self.EVENT_STUDY_GIVE_UP_DAYS:
                            # No price history / too little pre-event data.
                            # Record that it was tried so it is not retried.
                            self._persist_event_study({
                                "ticker": ticker, "event_id": event_id,
                                "direction": direction, "source_type": source_type,
                                "anchor": "disclosure", "anchor_date": disclosed,
                            }, now)
                except Exception as e:
                    logger.warning(
                        "orchestrator.event_study_failed", ticker=ticker, error=str(e)
                    )

            logger.info("orchestrator.event_study_done", measured=measured)
        except Exception as e:
            logger.warning("orchestrator.event_study_job_failed", error=str(e))

    @staticmethod
    def _return_series(price_data: dict) -> "pd.Series":
        """Daily log returns indexed by (naive, normalized) date."""
        series = pd.Series(
            np.asarray(price_data["returns"], dtype=float),
            index=FamaFrenchClient._to_naive_index(price_data["dates"][1:]),
        )
        return series[~series.index.duplicated(keep="last")]

    def _persist_event_study(self, result: dict, run_date: datetime):
        session = self.session_factory()
        try:
            es = EventStudyResult(
                ticker=result["ticker"],
                event_id=result["event_id"],
                run_date=run_date,
                direction=result["direction"],
                source_type=result.get("source_type"),
                car_1d=result.get("car_1d"),
                car_5d=result.get("car_5d"),
                car_10d=result.get("car_10d"),
                car_20d=result.get("car_20d"),
                t_statistic=result.get("t_statistic"),
                p_value=result.get("p_value"),
                is_significant=result.get("is_significant"),
                alpha=result.get("alpha"),
                beta=result.get("beta"),
                sigma_est=result.get("sigma_est"),
                n_estimation=result.get("n_estimation"),
                daily_cars=json.dumps(result["daily_cars"]) if result.get("daily_cars") else None,
                anchor=result.get("anchor"),
                anchor_date=result.get("anchor_date"),
            )
            session.add(es)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _pooled_signal_alpha(self) -> dict[tuple[str, str], dict]:
        """(source_type, direction) -> pooled post-disclosure history.

        Each value holds "event_study" (cluster-level CAR statistics) and
        "decay" (Bayesian decay fit of the mean abnormal-return curve). A
        group is absent until it has enough distinct disclosure dates.
        """
        session = self.session_factory()
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        try:
            rows = (
                session.query(EventStudyResult)
                .filter(EventStudyResult.anchor == "disclosure")
                .filter(EventStudyResult.car_20d.isnot(None))
                .all()
            )
            for r in rows:
                daily_ars = None
                if r.daily_cars:
                    try:
                        daily_ars = post_event_abnormal_returns(
                            json.loads(r.daily_cars), self.event_study.evt_start
                        )
                    except (ValueError, TypeError):
                        daily_ars = None
                groups[(r.source_type, r.direction)].append({
                    "anchor_date": r.anchor_date.date() if r.anchor_date else None,
                    "car_5d": r.car_5d,
                    "car_20d": r.car_20d,
                    "daily_ars": daily_ars,
                })
        finally:
            session.close()

        pooled = {}
        for key, items in groups.items():
            stats = pooled_event_stats(items)
            if stats is None:
                continue
            curve = stats.pop("mean_ar_by_day")
            fit = None
            if curve is not None:
                try:
                    fit = BayesianDecayAnalyzer().analyze(curve, direction=key[1])
                except Exception as e:
                    logger.warning("orchestrator.decay_fit_failed", group=key, error=str(e))
            pooled[key] = {"event_study": stats, "decay": fit}
            logger.info(
                "orchestrator.pooled_alpha",
                source=key[0], direction=key[1],
                n_events=stats["n_events"], n_clusters=stats["n_clusters"],
                mean_car_20d=stats["mean_car_20d"], p_value=stats["p_value"],
                half_life=fit.get("posterior_half_life") if fit else None,
            )
        return pooled

    def _persist_bayesian_decay(
        self, ticker: str, event_id: int, run_date: datetime, direction: str, fit: dict
    ):
        session = self.session_factory()
        try:
            strength = fit.get("signal_strength") or {}
            ci = fit.get("ci_half_life_95") or (None, None)
            session.add(BayesianDecayResult(
                ticker=ticker, event_id=event_id, run_date=run_date,
                direction=direction,
                n_days=fit.get("n_days"),
                total_car=fit.get("total_car"),
                posterior_half_life=fit.get("posterior_half_life"),
                posterior_lambda=fit.get("posterior_lambda"),
                ci_half_life_lower=ci[0],
                ci_half_life_upper=ci[1],
                entry_window_days=fit.get("entry_window_days"),
                exit_window_days=fit.get("exit_window_days"),
                annualized_ir=fit.get("annualized_ir"),
                decay_quality=fit.get("decay_quality"),
                signal_strength_5d=(strength.get(5) or {}).get("remaining_pct"),
                signal_strength_20d=(strength.get(20) or {}).get("remaining_pct"),
                prob_alpha_positive=fit.get("prob_alpha_positive"),
            ))
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    async def _weather_context(self) -> dict | None:
        """National weather context plus how unusual today's alert count is
        against the counts stored by previous cycles."""
        ctx = await self.weather.get_context()
        if not ctx:
            return ctx
        now = datetime.utcnow()
        history: list[float] = []
        session = self.session_factory()
        try:
            rows = (
                session.query(WeatherSnapshot)
                .filter(WeatherSnapshot.snapshot_date >= now - timedelta(days=BASELINE_WINDOW_DAYS + 1))
                .filter(WeatherSnapshot.n_severe_alerts.isnot(None))
                .all()
            )
            by_day: dict = defaultdict(list)
            for r in rows:
                if r.snapshot_date.date() < now.date():
                    by_day[r.snapshot_date.date()].append(r.n_severe_alerts)
            history = [float(np.mean(v)) for v in by_day.values()]
            session.add(WeatherSnapshot(
                snapshot_date=now,
                n_severe_alerts=ctx.get("n_severe_alerts"),
                temp_anomaly_c=ctx.get("temp_anomaly_c"),
                degree_day_delta=ctx.get("degree_day_delta"),
                n_cities=ctx.get("n_cities"),
            ))
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()
        return WeatherOverlay.with_alert_baseline(ctx, history)

    # Tickers needed before a cross-sectional options percentile means anything
    OPTIONS_MIN_REFERENCE = 20
    OPTIONS_MIN_OWN_HISTORY = 10

    def _options_reference(self) -> dict[str, list[float]]:
        """Latest put/call ratio and skew per ticker over the last 30 days,
        current method only — the population each ticker is ranked against."""
        session = self.session_factory()
        try:
            rows = (
                session.query(OptionsFlow)
                .filter(OptionsFlow.method_version == OPTIONS_METHOD_VERSION)
                .filter(OptionsFlow.analysis_date >= datetime.utcnow() - timedelta(days=30))
                .order_by(OptionsFlow.ticker, OptionsFlow.analysis_date.desc())
                .all()
            )
            latest: dict[str, OptionsFlow] = {}
            for r in rows:
                latest.setdefault(r.ticker, r)
            return {
                "pcr": [r.pcr for r in latest.values() if r.pcr is not None],
                "iv_skew": [r.iv_skew for r in latest.values() if r.iv_skew is not None],
            }
        finally:
            session.close()

    def _options_dict(self, session, opts_row, reference: dict[str, list[float]]) -> dict | None:
        """Options row -> ensemble input with percentile ranks attached."""
        if opts_row is None or opts_row.method_version != OPTIONS_METHOD_VERSION:
            return None  # legacy rows used a different, biased definition

        def rank(field: str, value):
            if value is None:
                return None
            ref = reference.get(field) or []
            if len(ref) < self.OPTIONS_MIN_REFERENCE:
                # Small cycle: fall back to the ticker's own recent history
                own = [
                    getattr(r, field) for r in session.query(OptionsFlow)
                    .filter(OptionsFlow.ticker == opts_row.ticker)
                    .filter(OptionsFlow.method_version == OPTIONS_METHOD_VERSION)
                    .filter(OptionsFlow.analysis_date >= datetime.utcnow() - timedelta(days=90))
                    .all()
                ]
                ref = [v for v in own if v is not None]
                if len(ref) < self.OPTIONS_MIN_OWN_HISTORY:
                    return None
            return percentile_rank(value, ref)

        return {
            "pcr": opts_row.pcr,
            "iv_skew": opts_row.iv_skew,
            "pcr_percentile": rank("pcr", opts_row.pcr),
            "iv_skew_percentile": rank("iv_skew", opts_row.iv_skew),
            "unusual_volume_score": opts_row.unusual_volume_score,
            "max_pain": opts_row.max_pain,
        }

    async def _run_copula_analysis(self, tickers: list[str]):
        """Run copula tail risk on each ticker vs market."""
        import json
        logger.info("orchestrator.copula_start", tickers=len(tickers))
        run_date = datetime.utcnow()

        ff_factors = await self.fama_french.get_factors(days=504)
        if ff_factors is None:
            logger.warning("orchestrator.copula_no_factors")
            return

        market_returns = ff_factors["Mkt-RF"].values

        for ticker in tickers:
            try:
                if not self._is_common_stock(ticker):
                    continue
                price_data = await self.yahoo.get_price_history(ticker, days=504)
                if price_data is None:
                    continue
                returns = price_data["returns"]
                return_dates = price_data["dates"][1:] if price_data.get("dates") else None
                aligned = self.fama_french.align_stock_to_factors(
                    returns, return_dates, ff_factors
                )
                if aligned is not None:
                    stock_aligned, factor_rows, _ = aligned
                    result = self.copula.analyze(
                        stock_aligned, factor_rows["Mkt-RF"].values
                    )
                else:
                    n = min(len(returns), len(market_returns))
                    result = self.copula.analyze(returns[-n:], market_returns[-n:])
                if result:
                    self._persist_copula(ticker, run_date, result)
            except Exception as e:
                logger.warning("orchestrator.copula_failed", ticker=ticker, error=str(e))

        logger.info("orchestrator.copula_done")

    def _persist_copula(self, ticker: str, run_date: datetime, result: dict):
        session = self.session_factory()
        try:
            row = CopulaTailRiskResult(
                ticker=ticker, run_date=run_date,
                n_observations=result["n_observations"],
                gaussian_rho=result["gaussian_rho"],
                student_t_rho=result["student_t_rho"],
                student_t_nu=result["student_t_nu"],
                tail_dep_lower=result["tail_dep_lower"],
                tail_dep_upper=result["tail_dep_upper"],
                joint_crash_prob=result["joint_crash_prob"],
                tail_dep_ratio=result["tail_dep_ratio"],
                var_95=result["var_95"],
                var_99=result["var_99"],
                cvar_95=result["cvar_95"],
                cvar_99=result["cvar_99"],
                conditional_var_95=result["conditional_var_95"],
                conditional_cvar_95=result["conditional_cvar_95"],
                tail_risk_score=result["tail_risk_score"],
            )
            session.add(row)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    # Largest universe handed to the optimizer. Beyond a few dozen names the
    # sample covariance from one year of data is mostly noise.
    MEANVAR_MAX_ASSETS = 30

    def _is_common_stock(self, ticker: str) -> bool:
        """False only when enrichment positively identified a non-common security."""
        return self._security_types.get(ticker) in (None, "common")

    def _portfolio_candidates(self, scored: list) -> list[str]:
        """Tickers of the highest-conviction BUY signals, best first."""
        buys = [
            (result.conviction, event.ticker)
            for event, _enrichment, result in scored
            if event.direction.value == "buy" and self._is_common_stock(event.ticker)
        ]
        tickers: list[str] = []
        for _conviction, ticker in sorted(buys, key=lambda b: -b[0]):
            if ticker not in tickers:
                tickers.append(ticker)
            if len(tickers) >= self.MEANVAR_MAX_ASSETS:
                break
        return tickers

    async def _run_mean_variance(self, tickers: list[str]):
        """Run mean-variance optimization across the given signal tickers."""
        import json
        import numpy as np

        if len(tickers) < 2:
            logger.info("orchestrator.meanvar_skip", reason="need >= 2 tickers")
            return

        logger.info("orchestrator.meanvar_start", tickers=len(tickers))
        run_date = datetime.utcnow()

        # Collect returns for all tickers
        import pandas as pd

        valid_tickers = []
        returns_list = []
        dates_list = []
        for ticker in tickers:
            try:
                price_data = await self.yahoo.get_price_history(ticker, days=252)
                if not price_data or len(price_data["returns"]) < 30:
                    continue
                # NaN/zero closes yield non-finite log returns, which make
                # LedoitWolf raise and would sink the whole optimization.
                if not np.all(np.isfinite(price_data["returns"])):
                    logger.warning("orchestrator.meanvar_nonfinite_returns", ticker=ticker)
                    continue
                valid_tickers.append(ticker)
                returns_list.append(price_data["returns"])
                dates = price_data.get("dates")
                dates_list.append(dates[1:] if dates else None)
            except Exception as e:
                logger.warning("orchestrator.meanvar_data_failed", ticker=ticker, error=str(e))

        if len(valid_tickers) < 2:
            logger.info("orchestrator.meanvar_skip", reason="insufficient valid tickers")
            return

        if all(d is not None and len(d) == len(r) for d, r in zip(dates_list, returns_list)):
            # Join on actual dates: tickers with a halt or a shorter listing
            # history would otherwise be shifted against the others.
            frame = pd.concat(
                [
                    pd.Series(r, index=FamaFrenchClient._to_naive_index(d), name=t)
                    for t, r, d in zip(valid_tickers, returns_list, dates_list)
                ],
                axis=1, join="inner",
            ).dropna()
            if len(frame) < 30:
                logger.info("orchestrator.meanvar_skip", reason="too few common trading days")
                return
            returns_matrix = frame.values
        else:
            # No dates available — align to common length
            min_len = min(len(r) for r in returns_list)
            returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])

        result = self.mean_variance.optimize(valid_tickers, returns_matrix)
        if result:
            self._persist_mean_variance(run_date, result)

        logger.info("orchestrator.meanvar_done")

    def _persist_mean_variance(self, run_date: datetime, result: dict):
        import json
        session = self.session_factory()
        try:
            row = MeanVarianceResult(
                run_date=run_date,
                n_assets=result["n_assets"],
                tickers=json.dumps(result["tickers"]),
                ms_weights=json.dumps(result["max_sharpe"]["weights"]),
                ms_return=result["max_sharpe"]["expected_return"],
                ms_volatility=result["max_sharpe"]["volatility"],
                ms_sharpe=result["max_sharpe"]["sharpe_ratio"],
                mv_weights=json.dumps(result["min_variance"]["weights"]),
                mv_return=result["min_variance"]["expected_return"],
                mv_volatility=result["min_variance"]["volatility"],
                ew_return=result["equal_weight"]["expected_return"],
                ew_volatility=result["equal_weight"]["volatility"],
                ew_sharpe=result["equal_weight"]["sharpe_ratio"],
                efficient_frontier=json.dumps(result["efficient_frontier"]),
                risk_contribution=json.dumps(result["risk_contribution"]),
            )
            session.add(row)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    async def _run_options_analysis(self, tickers: list[str]):
        """Fetch and analyze options chain data for each ticker."""
        for ticker in tickers:
            try:
                result = await self.options_client.get_options_analysis(ticker)
                if result:
                    self._persist_options_flow(result)
                    logger.debug("orchestrator.options_ok", ticker=ticker)
            except Exception as e:
                logger.warning("orchestrator.options_failed", ticker=ticker, error=str(e))
        logger.info("orchestrator.options_done", tickers=len(tickers))

    def _persist_options_flow(self, data: dict):
        session = self.session_factory()
        try:
            row = OptionsFlow(
                ticker=data["ticker"],
                analysis_date=data["analysis_date"],
                pcr=data.get("pcr"),
                unusual_volume_score=data.get("unusual_volume_score"),
                iv_skew=data.get("iv_skew"),
                max_pain=data.get("max_pain"),
                nearest_expiry=data.get("nearest_expiry"),
                total_call_volume=data.get("total_call_volume"),
                total_put_volume=data.get("total_put_volume"),
                total_call_oi=data.get("total_call_oi"),
                total_put_oi=data.get("total_put_oi"),
                method_version=data.get("method_version"),
            )
            session.add(row)
            session.commit()
        except IntegrityError:
            session.rollback()
        except Exception as e:
            session.rollback()
            logger.warning("orchestrator.options_persist_failed", error=str(e))
        finally:
            session.close()

    async def _run_ensemble_scoring(self, new_events: list):
        """Run ensemble scoring combining all model outputs per event."""
        logger.info("orchestrator.ensemble_start", events=len(new_events))
        run_date = datetime.utcnow()

        # Weather context: one national fetch per cycle, with alert baseline
        weather_ctx = None
        try:
            weather_ctx = await self._weather_context()
        except Exception as e:
            logger.debug("orchestrator.weather_ctx_failed", error=str(e))

        # Pooled post-disclosure history per (source, direction): the event
        # study and decay components of a fresh signal come from here.
        pooled_alpha: dict = {}
        try:
            pooled_alpha = self._pooled_signal_alpha()
        except Exception as e:
            logger.warning("orchestrator.pooled_alpha_failed", error=str(e))

        # Cross-section each ticker's options positioning is ranked against
        options_reference: dict = {}
        try:
            options_reference = self._options_reference()
        except Exception as e:
            logger.warning("orchestrator.options_reference_failed", error=str(e))

        # News sentiment: fetch once per unique ticker, persist, reuse
        news_cache: dict[str, dict | None] = {}

        for event_id, event_schema in new_events:
            try:
                ticker = event_schema.ticker
                direction = event_schema.direction.value

                if ticker not in news_cache:
                    news_result = None
                    try:
                        articles = await self.news_client.get_company_news(ticker, days=14)
                        news_result = self.news_analyzer.analyze(ticker, articles)
                        if news_result:
                            self._persist_news_sentiment(ticker, run_date, news_result)
                    except Exception as e:
                        logger.debug("orchestrator.news_failed", ticker=ticker, error=str(e))
                    news_cache[ticker] = news_result
                news_result = news_cache[ticker]

                session = self.session_factory()
                try:
                    # Gather latest model results for this ticker
                    from sqlalchemy import desc
                    mc = session.query(MonteCarloResult).filter(
                        MonteCarloResult.ticker == ticker
                    ).order_by(desc(MonteCarloResult.run_date)).first()

                    hmm = session.query(HMMRegimeState).filter(
                        HMMRegimeState.ticker == ticker
                    ).order_by(desc(HMMRegimeState.run_date)).first()

                    garch = session.query(GARCHForecast).filter(
                        GARCHForecast.ticker == ticker
                    ).order_by(desc(GARCHForecast.run_date)).first()

                    ff = session.query(FamaFrenchExposure).filter(
                        FamaFrenchExposure.ticker == ticker
                    ).order_by(desc(FamaFrenchExposure.run_date)).first()

                    copula_row = session.query(CopulaTailRiskResult).filter(
                        CopulaTailRiskResult.ticker == ticker
                    ).order_by(desc(CopulaTailRiskResult.run_date)).first()

                    opts_row = session.query(OptionsFlow).filter(
                        OptionsFlow.ticker == ticker
                    ).order_by(OptionsFlow.analysis_date.desc()).first()
                    opts_dict = self._options_dict(session, opts_row, options_reference)

                    enr_row = session.query(Enrichment).filter(
                        Enrichment.event_id == event_id
                    ).first()
                finally:
                    session.close()

                # Weather overlay: only contributes for weather-sensitive sectors
                weather_dict = None
                if weather_ctx and enr_row and enr_row.sector:
                    weather_dict = WeatherOverlay.score(
                        enr_row.sector, direction, weather_ctx
                    )

                # Convert DB rows to dicts for ensemble scorer
                mc_dict = {"horizons": {30: {
                    "probability_of_profit": mc.h30_prob_profit,
                    "expected_return": mc.h30_expected_return,
                }}} if mc else None

                hmm_dict = {"current_state": hmm.current_state,
                            "prob_bull": hmm.prob_bull, "prob_bear": hmm.prob_bear,
                            "prob_sideways": hmm.prob_sideways} if hmm else None

                garch_dict = {"forecast_5d_ratio": garch.forecast_5d_ratio,
                              "forecast_20d_ratio": garch.forecast_20d_ratio} if garch else None

                ff_dict = {"alpha_annual": ff.alpha_annual,
                           "alpha_tstat": ff.alpha_tstat,
                           "r_squared": ff.r_squared} if ff else None

                cop_dict = {"tail_risk_score": copula_row.tail_risk_score} if copula_row else None

                # Event study + decay: pooled history of comparable signals
                group = pooled_alpha.get((event_schema.source_type.value, direction)) or {}
                es_dict = group.get("event_study")
                decay_fit = group.get("decay")
                disclosed = event_schema.disclosure_date or event_schema.trade_date
                bd_dict = decay_component(
                    decay_fit, trading_days_since(disclosed, run_date)
                )
                if bd_dict:
                    self._persist_bayesian_decay(ticker, event_id, run_date, direction, decay_fit)

                # Earnings overlay
                earnings_dict = None
                try:
                    earnings_dict = await self.earnings_overlay.analyze(
                        ticker, event_schema.trade_date, direction
                    )
                except Exception as e:
                    logger.debug("orchestrator.earnings_failed", ticker=ticker, error=str(e))

                result = self.ensemble.score(
                    direction=direction,
                    monte_carlo=mc_dict,
                    hmm=hmm_dict,
                    garch=garch_dict,
                    fama_french=ff_dict,
                    copula=cop_dict,
                    bayesian_decay=bd_dict,
                    event_study=es_dict,
                    options_flow=opts_dict,
                    earnings_overlay=earnings_dict,
                    news_sentiment=news_result,
                    weather_overlay=weather_dict,
                )

                self._persist_ensemble(ticker, event_id, run_date, result)

            except Exception as e:
                logger.warning("orchestrator.ensemble_failed", ticker=event_schema.ticker, error=str(e))

        logger.info("orchestrator.ensemble_done")

    def _persist_ensemble(self, ticker: str, event_id: int, run_date: datetime, result: dict):
        session = self.session_factory()
        try:
            components = result.get("components", {})
            row = EnsembleScoreResult(
                ticker=ticker, event_id=event_id, run_date=run_date,
                direction=result["direction"],
                total_score=result["total_score"],
                confidence=result["confidence"],
                recommendation=result["recommendation"],
                n_models=result["n_models"],
                score_monte_carlo=components.get("monte_carlo"),
                score_hmm=components.get("hmm_regime"),
                score_garch=components.get("garch"),
                score_fama_french=components.get("fama_french"),
                score_copula=components.get("copula_tail"),
                score_bayesian_decay=components.get("bayesian_decay"),
                score_event_study=components.get("event_study"),
                score_options_flow=components.get("options_flow"),
                score_earnings_overlay=components.get("earnings_overlay"),
                score_news_sentiment=components.get("news_sentiment"),
                score_weather_overlay=components.get("weather_overlay"),
            )
            session.add(row)
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
        if self.news_client:
            await self.news_client.close()
        if self.edgar_xbrl:
            await self.edgar_xbrl.close()
        if self.edgar_13f:
            await self.edgar_13f.close()
        if self.macro_extra:
            await self.macro_extra.close()
        if self.notifier:
            await self.notifier.close()


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
