# Website Deployment Roadmap

Goal: the FastAPI backend + React dashboard running on a public domain,
pipeline running on schedule, with auth-ready architecture.

## Current state

- Backend: FastAPI (`src/api/api.py`), SQLite (`smart_money.db`), CORS pinned
  to localhost.
- Frontend: Vite + React in `dashboard/`, dev proxy `/api → localhost:8000`.
- Scheduler: GitHub Actions cron (7:00 + 17:30 ET) running the orchestrator
  with the DB cached between runs — works, but the website's DB and the
  pipeline's DB are currently separate worlds.
- Static demo: `reports/site/index.html` (rigour-first page for ericcosta.ca).

## Target architecture (cheap, boring, scalable enough)

```
GitHub Actions cron ──► orchestrator ──► Postgres (Supabase/Neon)
                                            ▲
Browser ──► Cloudflare Pages/Vercel        │
  (React static build)  └── /api/* ──► FastAPI on Fly.io/Railway ──┘
                                            │
                                       Slack webhook (alerts)
```

One Postgres instance becomes the single source of truth for both the
scheduled pipeline and the website API.

## Phase 1 — Make it deployable (1 week)

1. **SQLite → Postgres.** Already SQLAlchemy, so mostly config:
   - `DATABASE_URL=postgresql+psycopg://...` (add `psycopg[binary]` to reqs).
   - Replace SQLite-specific bits (none expected; verify `init_db`).
   - Free hosts: Supabase (500MB) or Neon (3GB). Either is plenty.
2. **Containerize the API:** small `Dockerfile` (python:3.11-slim, uvicorn,
   `--workers 2`). Healthcheck on `/`.
3. **Config hygiene:**
   - CORS origins from env var (`ALLOWED_ORIGINS`), not hard-coded localhost.
   - All secrets via env (already pydantic-settings — good).
4. **Frontend build:** `vite build` → static assets. Set API base URL by env
   (`VITE_API_URL`) instead of relying on the dev proxy.
5. **Point GitHub Actions at Postgres:** drop the DB cache steps, set
   `DATABASE_URL` secret. The pipeline then feeds the live site automatically.

## Phase 2 — Ship it (a weekend)

| Piece | Service | Cost |
|-------|---------|------|
| Frontend | Cloudflare Pages or Vercel | $0 |
| API | Fly.io (256MB machine) or Railway | $0–5/mo |
| Postgres | Supabase / Neon free tier | $0 |
| Domain | `app.ericcosta.ca` subdomain to start | $0 |
| Scheduler | existing GitHub Actions cron | $0 |
| Error tracking | Sentry free tier (FastAPI + React SDKs) | $0 |
| Uptime | UptimeRobot ping on `/` | $0 |

Total: **≤ $5/month** until real traffic.

Steps: deploy API container → run one pipeline cycle against Postgres →
deploy frontend with `VITE_API_URL` → wire DNS → smoke-test `/api/dashboard`,
`/api/top-pick`, `/api/sectors` → flip the GitHub Actions secret.

## Phase 3 — Public-readiness (before Phase 1 of the launch roadmap)

1. **Auth:** Supabase Auth or Clerk (magic links). FastAPI middleware validates
   the JWT; gate premium-only endpoints later.
2. **Rate limiting:** `slowapi` on the API (per-IP), Cloudflare in front.
3. **Caching:** the dashboard payload changes twice a day — cache
   `/api/dashboard`, `/api/sectors`, `/api/top-pick` for 5–15 min
   (in-process LRU or Cloudflare cache rules). Kills 95% of DB load.
4. **Backups:** Supabase/Neon have PITR on free/cheap tiers; also nightly
   `pg_dump` artifact in GitHub Actions.
5. **Migrations:** add Alembic before the schema starts evolving under users.
6. **Background jobs:** keep GitHub Actions as the only scheduler (free,
   observable, already hardened with pinned deps). Move to a worker dyno only
   if runs exceed the 15-min timeout.

## Phase 4 — Scale (only when needed)

- Read replicas / connection pooling (Supabase pooler, pgbouncer).
- Move heavy analysis off-request: it already is (pipeline writes, API reads).
- WebSocket push for live alerts (FastAPI native) — pairs with premium tier.
- CDN-cache the auto-generated SEO pick pages as static HTML.

## Order of operations checklist

- [ ] Provision Postgres, set `DATABASE_URL`, run `init_db`
- [ ] Dockerfile + deploy API to Fly/Railway
- [ ] Env-based CORS + `VITE_API_URL`
- [ ] Deploy dashboard to Cloudflare Pages/Vercel
- [ ] Repoint GitHub Actions cron at Postgres; delete DB cache steps
- [ ] DNS + HTTPS on app.ericcosta.ca
- [ ] Sentry + UptimeRobot
- [ ] First public morning pick lands on the live site automatically
