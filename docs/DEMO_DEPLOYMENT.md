# Deploy the finAlyse historical demo

This runbook deploys the React snapshot experience to Cloudflare Pages. The
frontend and reviewed JSON ship together. Python ingestion, model fitting, and
market-data requests run outside the visitor's session.

**Current delivery:** the September 16 implementation is being delivered through
GitHub at the user's request. No Cloudflare deployment is part of this release.
The portfolio CTA links explicitly to local setup instructions. The hosting steps
below are a future runbook, not an active deployment.

**Release record:** this document is a procedure, not proof that a Pages project,
deployment, or custom domain has been created. Record the actual deployment URL
and verified commit when publishing.

## 1. Verify the local release

From the repository root:

```bash
cd dashboard
npm ci
npm run build
npm run preview -- --host 127.0.0.1
```

Open the preview URL printed by Vite and complete the smoke test below. The
production artifact is `dashboard/dist/`.

Demo mode is the default. For an explicit deployment configuration, set
`VITE_DEMO_MODE=true`. The public demo must not require a running API, database,
API keys, or an admin token. The separate local research application uses
`npm run dev:live` with its FastAPI service running; `npm run build:live` builds
that interface. Those commands use `.env.live` to set `VITE_DEMO_MODE=false`.

The bundled JSON at `dashboard/public/demo/snapshot.json` is intended to be
public. The SQLite database and provider credentials are not deployment inputs.

## 2. Create the Pages project

Connect the GitHub repository in Cloudflare's **Workers & Pages → Create
application → Pages → Import an existing Git repository** flow. Use:

| Setting | Value |
| --- | --- |
| Repository | `ericnc09/financial-planner` |
| Project name | `finalyse-demo`, if available |
| Production branch | `main`, after the verified demo release is merged |
| Root directory | `dashboard` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Environment variable | `VITE_DEMO_MODE=true` for production and previews |

Save and deploy. Cloudflare installs dependencies and builds the project. Use
the actual `pages.dev` URL returned by the deployment; do not infer a URL from
the proposed project name. Future pushed commits and configured pull-request
previews can run the same build. See Cloudflare's [Vite deployment guide](https://developers.cloudflare.com/pages/framework-guides/deploy-a-vite3-project/)
and [build configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/).

The build consumes the checked-in snapshot. It must not generate fresh financial
data during a Pages build. That preserves the dataset reviewed for this release.

## 3. Verify the hosted preview

Run the smoke test against the deployment URL before using it in applications.
Record the result with the commit, snapshot ID, and date checked.

| Check | Expected result |
| --- | --- |
| First visit in a private window | The dataset loads without login or backend setup. |
| Historical label | Snapshot date is visible; no current or live-data claim. |
| Search and filters | Matching disclosures appear; no-match combinations explain the empty result. |
| Open an example | Actor, direction, original event dates, and available source information appear. |
| Inspect research | Available outputs render; absent data is explicitly unavailable. |
| Source link | The destination corresponds to its label; provider pages are not presented as a verified original filing if that URL is absent. |
| Model wording | Experimental output, model coverage, and data limitations are distinguishable. |
| Refresh and shared links | Reloading the app and any supported selection URL still works. |
| Keyboard navigation | Controls and disclosures can be reached and activated with visible focus. |
| Small screen | At roughly 390 px width, primary actions and evidence remain readable without page-wide clipping. |
| Network activity | Core exploration uses static assets; there are no requests to `/api`, localhost, or live market-data services. |
| Load failure | Blocking the snapshot request shows a useful failure state with a retry path. |
| Console | No uncaught errors during the core journey. |

The demo does not promise price history where none was saved. A simulation chart
must be labeled as simulated research output, never as an observed price chart.

## 4. Add the domain and portfolio handoff

The proposed domain is `finalyse.ericcosta.ca`.

1. In the verified Pages project, open **Custom domains → Set up a domain** and
   enter the proposed subdomain.
2. Follow Cloudflare's DNS instructions. With an external DNS provider, create
   the indicated CNAME pointing to the project's actual `pages.dev` hostname.
3. Wait for domain activation and HTTPS, then repeat the smoke test on the
   custom domain.
4. Set the existing portfolio page's primary **Try the demo** button to that
   verified URL. Keep GitHub as the secondary action.
5. Apply the corrected copy in [FINTECH_PM_CASE_STUDY.md](FINTECH_PM_CASE_STUDY.md),
   then check the portfolio-to-demo journey in a private browser window.
6. Add the verified demo URL near the top of the repository README and its
   GitHub About website field.

Associate the custom domain with Pages before manually creating its DNS record.
A subdomain can use an external DNS provider. See Cloudflare's [custom domain
instructions](https://developers.cloudflare.com/pages/configuration/custom-domains/).

The portfolio website is a separate source/deployment in
[ericnc09/ericcosta_portfolio](https://github.com/ericnc09/ericcosta_portfolio).
Its canonical case-study source is
`src/routes/insights.stocks.smart-money-signals.tsx`; project-card and legacy
case-study content live in `src/lib/site.ts` and `src/lib/case-studies.tsx`.
`FINALYSE_DEMO_URL` in `src/lib/site.ts` sets the destination across those pages.
Editing the copy in this repository does not update `ericcosta.ca`.

The portfolio uses TanStack Start and a Cloudflare Worker build, separate from
the static Pages demo. In that repository, install its locked dependencies with
`bun install --frozen-lockfile`, then run `npm run build`. The verified build
produces `dist/client`, `dist/server`, and a generated
`dist/server/wrangler.json`; use the existing portfolio deployment integration
and verify its production route after publishing.

## 5. Refresh the snapshot deliberately

The first demo uses a June 22, 2026 research boundary. The release date and data
date are different and must remain distinct.

From the repository root, export from the local research database:

```bash
python3 scripts/export_demo_snapshot.py \
  --as-of 2026-06-22 \
  --database smart_money.db \
  --audit reports/model_status_report.json \
  --output dashboard/public/demo/snapshot.json
```

The exporter uses read-only SQLite access and validates the payload before an
atomic replacement of the destination. It needs only Python's standard library.
It preserves original event/model dates and includes provenance and limitations.
The initial export excludes enrichment records whose ticker associations were
not sufficiently reliable for the public snapshot.

For a later release:

1. Correct relevant ingestion/data-integrity findings before refreshing the
   source research database. Do not run upstream ingestion merely to change the
   date on an otherwise unchanged demo.
2. Choose a new explicit research boundary only when the available source data
   supports it. Generate the associated research audit separately and preserve
   its actual timestamp.
3. Run the exporter on a review branch with that boundary and audit. The
   checked-in previous snapshot remains available through Git.
4. Review the exported diff: selected tickers, event counts, source URLs, dates,
   missing components, public-field selection, and limitations. Do not silently
   replace missing observed data with synthetic examples.
5. Run validation and build:

   ```bash
   .venv/bin/python -m pytest tests/test_demo_snapshot.py -q
   npm run --prefix dashboard build
   ```

6. Run the hosted smoke test on the branch preview. Release the frontend and
   snapshot together only after they work as a pair.
7. Record the production URL, deployment ID, commit, snapshot ID, data boundary,
   verification date, and any changed limitations.

A failed model-validation gate is research evidence that must be disclosed.
A malformed snapshot, broken visitor journey, or unsupported public claim is a
demo release defect that needs correction before publication.

## 6. Roll back a bad release

1. Open **Deployments** in the Pages project and identify the last production
   deployment that passed the smoke test.
2. Use its actions menu to choose **Rollback to this deployment** and confirm.
   Cloudflare allows successful production deployments as rollback targets;
   preview deployments are not eligible.
3. Repeat the smoke test on the production domain and confirm the expected
   snapshot date.
4. Revert the faulty source/snapshot change through the normal review flow so
   the next automatic deployment does not reintroduce it. Keep the failed
   deployment and its diagnostic evidence available for investigation.

Source: Cloudflare's [rollback procedure](https://developers.cloudflare.com/pages/configuration/rollbacks/).

## Release note template

Copy this into the release description and fill it from the verified build:

```text
Demo URL:
Pages deployment ID:
Git commit:
Snapshot ID:
Data boundary (as_of):
Snapshot generation time:
Research audit timestamp:
Verified on:
Checks completed:
Known limitations:
Previous working production deployment:
```

The first release demonstrates a usable research workflow. User research,
adoption, retention, and investment-performance results remain separate evidence
to collect and report accurately.
