#!/bin/sh
# Run the pipeline once. Called by cron, which provides a minimal environment,
# so everything is resolved from this script's location.
# Settings, including alert destinations, come from config/.env.

cd "$(dirname "$0")/.." || exit 1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# A long backlog run can outlast the gap between scheduled slots, and two runs
# writing to the same SQLite database would conflict.
if pgrep -f "src.pipeline.orchestrator" > /dev/null; then
    echo "$(timestamp) run_pipeline: skipped, another pipeline run is in progress"
    exit 0
fi

echo "$(timestamp) run_pipeline: starting"
.venv/bin/python -m src.pipeline.orchestrator --mode oneshot
status=$?
echo "$(timestamp) run_pipeline: finished with exit code $status"
exit "$status"
