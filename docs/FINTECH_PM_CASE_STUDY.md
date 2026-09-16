# finAlyse: making equity research easier to inspect

**An independent product prototype by Eric Costa.**

Public trading disclosures contain useful research leads, but a transaction
alone does not explain its significance. finAlyse brings a disclosure, its market
context, and the limits of the available analysis into one research workflow.

**Prototype scope:** explore a dated research snapshot, inspect supporting
evidence, and understand what remains uncertain.

[Explore the code](https://github.com/ericnc09/financial-planner) ·
[Read the technical review](FINTECH_PM_DEMO_REVIEW.md)

## The problem I am testing

**Audience hypothesis:** self-directed investors who already research individual
stocks and want to understand public trading disclosures.

**Job hypothesis:** “When I encounter a trading disclosure, help me decide what
deserves further research by showing the source, timing, supporting context,
and unanswered questions.”

This is a product hypothesis. No user interviews, adoption results, or validated
market demand are claimed in this case study. The next step is to test this
workflow with people who match the proposed audience.

## My role and approach

I own the project and the decision to narrow its portfolio demo to explainable
disclosure research. The repository connects data ingestion, storage, Python
analysis, a FastAPI service, and a React interface. For this release, I chose the
research journey, the evidence to show, and the tradeoffs between breadth,
reliability, and understandable output. The September 2026 review and demo
implementation use Codex assistance.

This is an independent prototype, with no claimed employer, client engagement,
customer team, or commercial launch. Implementation evidence lives in the
repository; product assumptions remain open to testing.

## What the review established

The September 2026 review identified a gap between the breadth of the research
code and the experience a first-time visitor could use confidently:

- The existing portfolio page sent “Start Analyzing” to GitHub.
- Date filters anchored to the current day made the saved historical database
  produce an empty default dashboard.
- The website displayed performance figures without visible supporting
  evidence. The local model audit did not establish those public claims.
- Several analytical components were implemented but lacked populated results
  in the saved database.

These are product and repository observations from the review, not findings
from user interviews. They made the immediate priority clear: deliver one
complete research journey with transparent evidence.

## The MVP decision

The demo centers on three tasks:

1. **Find a disclosure.** Explore a bounded set of historical examples using
   filters and select an event.
2. **Inspect its evidence.** Read the transaction and disclosure dates, follow
   its source, and inspect the analysis available for that ticker.
3. **Explain its limits.** Identify missing information and distinguish reported
   facts from experimental model outputs.

The historical snapshot keeps original dates. Examples are selected for usable
data coverage, including incomplete cases; selection is not a claim that these
stocks represent superior investment opportunities.

## Three product decisions

### 1. Make timing and provenance part of the research journey

A transaction date and a disclosure date answer different questions. The
interface needs both so readers can distinguish when an event happened from
when it became available to research. Sources and snapshot dates belong beside
the evidence they explain.

**Tradeoff:** this adds information to the interface. Plain-language summaries
and expandable detail help readers inspect that information at their own pace.

**What would change my decision:** if usability sessions show that readers
cannot identify the source or available-at date, improve those cues before
adding more models.

### 2. Explain coverage before promoting scores

A model score is an experimental research output. It does not establish the
probability that a trade will succeed. Likewise, model coverage describes how
much of the analysis is populated; it is not predictive confidence.

The saved model report dated June 22, 2026 contains 202 events, 189 enrichments,
and 189 latest ensemble rows. It reports three of seven validation gates passing
and an untrained XGBoost artifact. The full report covers the research database;
the demo presents a selected subset.

**Tradeoff:** readers need to understand coverage and limitations before
interpreting a score. Concise labels and expandable detail keep that explanation
close to the relevant output.

**What would change my decision:** stronger point-in-time evaluation and useful
results against a defined baseline could justify additional analytical claims.
Passing the present gates alone is insufficient: the review also identified
historical-evaluation issues that need correction.

Evidence: [saved model report](../reports/model_status_report.json) and
[evaluation findings](FINTECH_PM_DEMO_REVIEW.md#5-technical-findings-and-priority).

### 3. Make the first visit independent of data-provider availability

The portfolio demo uses bundled data reviewed before release. Visitors can
filter and inspect that dataset without starting the Python pipeline or waiting
for a market-data request. The deployed artifact includes both the interface
and its snapshot, making releases reproducible and recoverable.

**Tradeoff:** visitors see historical research with an explicit date. Fresh data
and arbitrary server-side analysis remain future work.

**What would change my decision:** evidence that target users return regularly
and need updated research would justify a shared database, scheduled refreshes,
and a production API.

## Results and what remains unproven

| Evidence type | What can be said |
| --- | --- |
| Existing implementation | The repository contains ingestion, persistence, analysis, API, and frontend code. |
| Saved research audit | The June 22 report documents component coverage, validation failures, and an untrained model. |
| Demo scope | A focused historical research journey is the chosen first release. Its release verification belongs with the deployed build. |
| Usability outcomes | Not measured yet. |
| User demand and retention | Not measured yet. |
| Investment effectiveness | Not established by this demo or its selected examples. |

The product lesson so far is about prioritization: a broad analysis stack is
useful only when a reader can connect its output to evidence and understand
what the output cannot tell them.

## The next experiment

Run five to eight task-based sessions with people who already research
individual equities. Give each participant one disclosed transaction and ask:

> Explain what happened, when the information became available, and what you
> would need to check before drawing an investment conclusion.

Have participants attempt a comparable task with their usual tools and with the
prototype. Alternate the task order to reduce practice effects. Ask them to
think aloud, record where they hesitate, and obtain their consent before any
recording. This small study would provide directional usability evidence.

| Proposed measure | How to collect it | Decision it informs |
| --- | --- | --- |
| Research-task completion | Count participants who identify the actor, direction, disclosure timing, and source without assistance. | Whether the core journey is understandable. |
| Time to a sourced explanation | Measure from task start to the participant's first complete explanation. | Where navigation or presentation adds friction. |
| Understanding uncertainty | Ask what model coverage means and what the prototype cannot conclude. | Whether labeling needs revision. |
| Misinterpretation | Record any treatment of a score as a guaranteed outcome or current recommendation. | Whether to reduce score prominence or change explanations. |
| Desire to reuse | Ask when this workflow would fit a real research task and invite a later follow-up. | Whether to prioritize freshness or deepen the workflow. |

These are proposed measures. There are no measured improvements or target
achievement claims yet. After the sessions, publish actual counts, representative
observations, limitations, and the resulting prioritization decision.

## Portfolio page replacement copy

Use this copy on the existing portfolio page once the demo URL is verified:

**Title:** finAlyse — explainable equity research

**Subtitle:** Explore public trading disclosures, supporting evidence, and the
limitations of experimental models in a historical research demo.

**Primary button:** Try the demo

**Secondary button:** Explore the code

**Status label:** Historical research prototype

**Short project description:**

> I am developing finAlyse to explore a product question: can an evidence-first
> workflow help people interpret trading disclosures more clearly? The demo
> brings source records and available model context together, labels missing
> information, and keeps historical dates visible. Its next milestone is
> task-based usability research with the proposed audience.

**Three supporting cards:**

- **Trace the evidence.** Inspect the actor, transaction, disclosure timing, and
  source behind an example.
- **Understand the limits.** See which research outputs are available and which
  questions the current data cannot answer.
- **Explore the product decisions.** Read how scope, reliability, and financial
  uncertainty shaped the MVP.

Remove unsupported live, return, win-rate, Sharpe, predictive-confidence, and
institutional-quality claims when applying this copy. A button should open its
named destination; add the live demo URL only after that deployment is verified.
The deployment procedure is in [DEMO_DEPLOYMENT.md](DEMO_DEPLOYMENT.md).
