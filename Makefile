.PHONY: init-db run-once run-daemon run-api run-dashboard test backtest install

install:
	pip install -r requirements.txt

init-db:
	python -m src.models.database

run-once:
	python -m src.pipeline.orchestrator --mode oneshot

run-daemon:
	python -m src.pipeline.orchestrator --mode daemon

run-api:
	uvicorn src.api.api:app --reload --port 8000

run-dashboard:
	cd dashboard && npm run dev

test:
	pytest tests/ -v

backtest:
	python -m src.backtesting.backtester --start 2024-01-01 --end 2025-12-31
