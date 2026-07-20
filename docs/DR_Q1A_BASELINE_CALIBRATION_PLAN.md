# DR-Q1A — Deep Research baseline calibration plan

Status: **Owner-reviewable plan, NOT yet executed, NOT yet
authorized for execution.** This document describes the protocol
that, once approved, will measure the current production Deep
Research pipeline at `b95afb4943a855eb0cc4fdd911218bbf0d6087b6`
before any further quality intervention.

This plan is the second outcome of the DR-Q0.1 capability-truth
synchronization. The first outcome is `docs/CAPABILITY_LEDGER.md`
updated to reflect the post-Slice-1C3 reality. Both outcomes are
documentation-only. Neither modifies the production pipeline.

## 1. Purpose

DR-Q1A measures the existing Deep Research pipeline against a
private, frozen, versioned 24-case benchmark with structured human
review. The goal is to establish a baseline of the current pipeline's
observed quality, identify the dominant observed failure mode (if
any), and let that failure (or the absence of a dominant failure)
select the next authorized experiment.

Calibration is a measurement, not a modification. The pipeline MUST
NOT change during the calibration pilot.

## 2. Non-goals

This slice and this calibration explicitly do NOT include:

- any production change to `hermes/`;
- any change to the test suite in `tests/`;
- any new schema, migration, or DB index;
- any new dependency, package, or framework;
- any benchmark execution harness implementation (only the plan
  lives in the repository; the harness, if any, lives outside
  this slice and outside this PR);
- any executable benchmark fixture (no JSON, no SQL, no Python
  fixtures, no harness);
- any benchmark execution at all in this slice;
- any LLM-as-judge automation;
- any claim parser, claim verifier, evidence ledger, or
  citation-support checker;
- any iterative retrieval, multi-pass planning, reflection step,
  re-query logic, or stopping decision;
- any query decomposition (static or learned);
- any contradiction handling;
- any source deduplication or authority policy;
- any new prompts in production;
- any change to the daily or per-job budget;
- any modularization, worker separation, or process separation;
- any provider spending beyond what the owner explicitly approves
  in a later authorization;
- any access to private user data, credentials, or personal cases;
- any deployment, release, or production rollout;
- any claim that the existing pipeline is "production ready",
  "fully sovereign", "frontier quality", or "competitive with
  Perplexity" or any other commercial system.

## 3. Baseline freeze contract

Before the calibration pilot can begin, the following fields MUST be
frozen and recorded in the run manifest. No field may change
mid-pilot. Any change requires a new owner authorization and a new
pilot.

| Field | Freeze at |
| --- | --- |
| Repository commit | `b95afb4943a855eb0cc4fdd911218bbf0d6087b6` (post-1C3) |
| Operating mode | `HERMES_DEEP_RESEARCH_ENABLED=true` (opt-in) |
| Selected model identifier | Owner-approved in a later authorization |
| Selected provider | Owner-approved in a later authorization |
| Selected search backend | Owner-approved in a later authorization |
| Fetch policy | The reviewed `SafeExternalFetcher` policy at `b95afb4` |
| Maximum sources | `settings.deep_research_max_sources` as frozen at startup |
| Output limits | `deep_research_per_source_max_tokens`, `deep_research_output_max_tokens` as frozen at startup |
| Prompts | The prompts in `hermes/jobs/prompts.py` at `b95afb4` |
| Daily budget | `deep_research_daily_budget_usd` as frozen at startup |
| Per-job budget | Recorded but NOT enforced as a hard cancellation (soft warning only) |
| Execution date | The date the pilot is run |
| Execution environment | Owner-approved |
| Dependency lock | `requirements-ci.lock` at `b95afb4` |
| Evaluator rubric version | `rubric-v0.1-draft` (see Section 9) |
| Case corpus version | `corpus-v0.1-draft` (see Section 4) |
| Reviewer workflow | Structured manual review (see Section 10) |
| Handling of resulting reports | Stored under the owner's chosen location outside this repository; never published without owner approval |

No secret values appear in this list. The run manifest stores
configuration FINGERPRINTS, not values.

## 4. Candidate corpus

The proposed candidate corpus contains **24 cases** spread across
**8 families**, with **3 cases per family**. The corpus is
documentary only at this stage; no executable fixture is created
in this slice.

### Families

1. **Recent factual research** — questions whose primary sources
   are dated within the last 12 months, with a clear "evaluated_at"
   and a "stale_after" that bounds the freshness expectation.
2. **Technical architecture** — questions whose primary sources are
   documentation, RFCs, and well-known reference works. Expected
   citation count moderate; expected contradiction rate low.
3. **Product comparison** — questions whose primary sources are
   vendor documentation and independent benchmarks. Expected
   citation count high; expected bias-detection difficulty high.
4. **Regulation and official guidance** — questions whose primary
   sources are regulator pages, official PDFs, and government
   databases. Expected citation count low; expected authority check
   easy.
5. **Travel and logistics** — questions whose primary sources are
   official transport / accommodation pages, schedules, and
   policies. Expected citation count moderate; expected volatility
   high (schedules change).
6. **Contradictory reliable sources** — questions for which two or
   more authoritative sources disagree. Expected contradiction
   detection difficulty high; expected value of structured manual
   review high.
7. **Multi-branch research** — questions whose answer requires
   following more than two independent branches (e.g. compare
   policy in three jurisdictions, or summarize a 5-year timeline).
   Expected coverage difficulty high; expected value of static
   decomposition highest in this family.
8. **Legitimate uncertainty** — questions whose answer is not
   fully knowable from public sources at the time of the question
   (e.g. near-future forecasts, rapidly changing commercial terms,
   emerging technical standards). Expected "claim extraction"
   difficulty high; expected value of an explicit "unknown"
   disposition high.

### Selection rules

- No personal or sensitive information.
- No medical diagnosis, no personalized financial advice, no legal
  advice for a specific user.
- No case that requires illegal access, credential bypass, or
  scraping a private surface.
- Avoid questions with one fragile exact answer (e.g. "what is the
  exact population of city X as of YYYY-MM-DD"). Such questions
  conflate freshness with quality.
- Prefer cases that can be audited from primary sources that the
  reviewer can fetch independently.
- Mark every volatile fact with `evaluated_at` and `stale_after`.
  Cases whose answer changed between `evaluated_at` and the pilot
  execution date are flagged for the reviewer.
- Do not fabricate gold answers for rapidly changing cases. The
  reviewer judges the process, not a single answer.
- Prevent the corpus from becoming a trivia benchmark by including
  questions that require synthesis across multiple sources.
- Include questions representative of a real personal AI
  assistant: "should I", "compare", "what changed", "is X safe",
  "summarize the past year of", "plan a trip to".

### 24 cases (3 per family, documentary only)

| case_id | family | draft prompt | freshness |
| --- | --- | --- | --- |
| rec-01 | Recent factual | What were the three most-cited peer-reviewed papers on long-context LLM evaluation published in 2025? | stale_after=2026-12-31 |
| rec-02 | Recent factual | What is the current (2025) state of the EU AI Act enforcement actions against foundation-model providers? | stale_after=2026-06-30 |
| rec-03 | Recent factual | What are the most recent 12 months of CVE-class RCE vulnerabilities disclosed in major open-source routers? | stale_after=2026-12-31 |
| arch-01 | Technical architecture | Compare the architectures of PostgreSQL's MVCC implementation and FoundationDB's record-layer implementation. | low |
| arch-02 | Technical architecture | Explain how Rust's borrow checker handles async closures. Cite the relevant language reference sections. | low |
| arch-03 | Technical architecture | How does the Linux kernel's cgroup v2 freezer interact with systemd-managed services? | low |
| prod-01 | Product comparison | Compare the feature sets, pricing models, and self-hosting options of three password managers targeting small teams. | moderate |
| prod-02 | Product comparison | Compare the OCR engines Tesseract, PaddleOCR, and Surya for historical document transcription. | low |
| prod-03 | Product comparison | Compare the local-first note-taking apps Obsidian, Logseq, and Anytype across data ownership, sync, and plugin models. | low |
| reg-01 | Regulation | Summarize the data-residency requirements for clinical data under HIPAA and the GDPR. | low |
| reg-02 | Regulation | What is the current process for filing a security advisory with the Python Security Response Team? | moderate |
| reg-03 | Regulation | Summarize the European Accessibility Act requirements for e-commerce sites, with primary sources. | low |
| travel-01 | Travel and logistics | Compare the standard checked-baggage allowance and weight limits for the three major transatlantic airline alliances. | stale_after=2026-12-31 |
| travel-02 | Travel and logistics | What is the standard visa-on-arrival policy for a European passport holder entering each of the four largest ASEAN countries? | stale_after=2026-12-31 |
| travel-03 | Travel and logistics | Summarize the current pet-import requirements for cats and dogs entering the UK from the EU. | stale_after=2026-12-31 |
| contra-01 | Contradictory sources | Summarize the current evidence on the effect of intermittent fasting on insulin resistance, citing primary sources that disagree. | moderate |
| contra-02 | Contradictory sources | What does the evidence say about coffee consumption and cardiovascular risk? Surface the disagreement. | moderate |
| contra-03 | Contradictory sources | How effective is static typing at preventing bugs in large codebases? Surface both sides. | moderate |
| multi-01 | Multi-branch | Plan a 7-day trip to Japan in late October for a vegetarian family of four on a moderate budget. Compare JR Pass options. | low |
| multi-02 | Multi-branch | Summarize the past 5 years of the Python packaging story (PEP 517, PEP 518, PEP 621, pyproject.toml, uv, hatchling). | low |
| multi-03 | Multi-branch | Compare the data-protection regimes of Brazil (LGPD), California (CCPA/CPRA), and the EU (GDPR) for a small SaaS company. | low |
| uncert-01 | Legitimate uncertainty | Predict the most likely 2027 standardization outcome for the W3C Web Neural Network API. | explicit "unknown" disposition required |
| uncert-02 | Legitimate uncertainty | Forecast the 2026 price of memory (DDR5) given current supply-chain signals. | explicit "unknown" disposition required |
| uncert-03 | Legitimate uncertainty | Is it safe to run a 5kW continuous load on a 15A residential circuit in North America? | explicit "unknown" disposition required |

No personal data. No medical diagnosis. No personalized financial
advice. No illegal access. The 6 freshness-volatile cases (the 3
recent-factual cases and the 3 travel-and-logistics cases) carry
`stale_after` annotations; the 3 legitimate-uncertainty cases
carry an explicit "unknown" disposition requirement. The
`evaluated_at` value for every case is recorded in its populated
case sheet, not in this summary table.

## 5. Case-sheet schema

Every case in the corpus carries the following documentary fields:

| Field | Meaning |
| --- | --- |
| `case_id` | Stable identifier (e.g. `rec-01`). |
| `family` | One of the 8 family names above. |
| `draft_prompt` | The literal user prompt. |
| `user_intent` | Short description of the user's actual need. |
| `expected_subquestions` | Optional list of sub-questions the answer should cover. |
| `expected_primary_source_types` | What kind of sources we expect (e.g. regulator page, vendor docs, RFC). |
| `critical_claims_to_inspect` | The claims the reviewer MUST verify by hand. |
| `unacceptable_failures` | What would cause this case to score 0 (e.g. fabricated citation). |
| `freshness_level` | `low` / `moderate` / `volatile`. |
| `evaluated_at` | The date the case was last audited. |
| `stale_after` | The date after which the case must be re-audited before running. |
| `expected_difficulty` | `easy` / `moderate` / `hard` (reviewer's qualitative sense). |
| `estimated_research_cost_class` | `low` (~$0.01-$0.05), `medium` (~$0.05-$0.20), `high` (~$0.20-$0.50). |
| `owner_notes` | Free text. |
| `approval_state` | `proposed` / `owner_review_required` / `frozen` / `rejected`. |

The schema is documentary only. No executable fixture, no JSON
file, no SQL row, and no Python constant is created in this slice
or in the calibration slice. The case sheets will be the input
to a future owner-approved pilot run, not to this slice.

The summary table in §4 is a condensed view; every field above
lives in the populated case sheet for each case, not in the
table. In particular, `evaluated_at` and `stale_after` are
populated per case sheet when the corpus moves from
`approval_state = owner_review_required` to `frozen`, and the
table inherits the relevant freshness annotations only for the
cases that need them at review time. The selection rule that
volatile facts must carry both `evaluated_at` and `stale_after`
is enforced at case-sheet freeze time, not at table-summary
time.

## 6. Eight-case pilot proposal

The pilot proposes **one candidate per family**. All eight are
marked `OWNER REVIEW REQUIRED / NOT YET FROZEN / NOT AUTHORIZED
FOR EXECUTION`. None of them is executed in this slice or in the
calibration slice until the owner approves the full pilot.

| case_id | Family | Why useful | Failure mode it can reveal |
| --- | --- | --- | --- |
| rec-01 | Recent factual | A recent factual question with primary sources; a reviewer can verify citations in minutes. | Fabrication of citations; stale information; authority miscalibration. |
| arch-01 | Technical architecture | A non-volatile technical question; a reviewer can verify against language references. | Misinterpretation of a technical concept; missing primary source (RFC, language reference). |
| prod-01 | Product comparison | A comparison question requires synthesis across multiple vendor sources. | Source bias; missing alternatives; undated information. |
| reg-01 | Regulation | A regulation question with very high authority signal. | Fabricated authority; missing the official source. |
| travel-01 | Travel and logistics | A volatile question with `stale_after`. | Stale information; failure to surface the freshness caveat. |
| contra-01 | Contradictory sources | A question with a documented disagreement. | Silent merging of contradictions; false certainty. |
| multi-01 | Multi-branch | A multi-branch question with explicit constraints. | Coverage failure; one-branch-only answer. |
| uncert-01 | Legitimate uncertainty | A question whose answer is not fully knowable. | False certainty; missing "unknown" disposition. |

These eight cases are intentionally diverse: each one targets a
specific failure mode that the calibration wants to make
observable. Together they cover all 8 families.

The pilot explicitly excludes any personal, medical, or
personalized-financial case. No provider spending happens in this
slice.

## 7. Repetition protocol

The pilot proposes:

- 8 pilot cases
- x 2 independent executions
- = **16 total research jobs**

Both runs use the SAME frozen configuration. The repetition is
NOT to average away failures; it is to estimate variance. A
research job that produces a "good" report on run 1 and a "bad"
report on run 2 (or vice versa) is a HIGH-VARIANCE finding, not a
PASS. The rubric will record this explicitly.

The 16 research jobs are NOT executed in this slice. They are
described here as the protocol that the owner will approve in a
later authorization.

## 8. Run manifest

Every future run of the calibration pilot (NOT in this slice) will
record the following fields per research job. The fields are
documentary. No values are recorded in this slice.

| Field | Meaning |
| --- | --- |
| `case_id` | Which case (e.g. `rec-01`). |
| `run_id` | Unique run identifier (e.g. `dr-q1a-rec-01-r1`). |
| `job_id` | The Deep Research job_id produced by `POST /v1/jobs`. |
| `git_commit` | The exact repository commit at the time of execution. |
| `configuration_fingerprint` | A short hash of the frozen configuration, never the values. |
| `model_identifier` | The model the service actually called. |
| `provider_identifier` | The provider the service actually called. |
| `search_backend` | The search backend the service actually called. |
| `started_at` | Timestamp the job was submitted. |
| `completed_at` | Timestamp the job reached a terminal state. |
| `total_latency_s` | Wall-clock duration. |
| `tokens_in` | Sum of input tokens across all LLM calls. |
| `tokens_out` | Sum of output tokens across all LLM calls. |
| `cost_usd` | Recorded per-job cost. |
| `final_status` | `complete` / `failed` / `cancelled`. |
| `source_urls` | The list of source URLs the service actually fetched. |
| `report_artifact_ref` | A reference (NOT a path) to the report artifact. |
| `retry_information` | Any retry or recovery events. |
| `reviewer` | The handle of the human reviewer. |
| `rubric_version` | The rubric version used. |

No credentials, no API keys, no raw filesystem paths, and no
internal exception text appear in the manifest. Sensitive values
stay under the owner's chosen storage.

## 9. Draft quality rubric

The draft rubric uses a **0-3 integer scale** per dimension. The
owner will calibrate thresholds after the pilot.

| Dimension | What is judged | 0 anchor | 1 anchor | 2 anchor | 3 anchor | Unacceptable failure | Evidence | Ambiguity notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Factual accuracy | Are the verifiable claims correct? | Multiple false or fabricated claims. | Some correct, some unverifiable. | Mostly correct, minor inaccuracies. | All verifiable claims correct. | Fabricated citation. | Manual: reviewer fetches the cited page and checks. | "Correct" is judged on the cited source, not on the reviewer's prior knowledge. |
| Completeness | Does the answer cover the expected subquestions? | Major branches missing. | One or more branches partially covered. | Most branches covered. | All expected branches covered. | A whole branch missing AND no acknowledgement. | Manual: reviewer checks the expected_subquestions. | For multi-branch cases, "covered" means at least one citation per branch. |
| Recency | Are sources recent enough for the question? | Sources are all older than `stale_after`. | One source is recent; others are stale. | Most sources are recent. | All sources are recent enough. | Sources older than `stale_after` AND no caveat. | Manual: reviewer checks the source date. | "Recent" is relative to the `stale_after` field, not the current date. |
| Source authority | Are the sources authoritative for the question? | Sources are low-authority for the question. | Mixed authority. | Mostly authoritative. | Sources are the canonical authority for the question. | A primary source is missing AND no acknowledgement. | Manual: reviewer compares to `expected_primary_source_types`. | "Authoritative" is judged per-family, not globally. |
| Source diversity and deduplication | Are sources distinct and from different domains? | Multiple duplicated or syndicated sources. | Some duplication, some diversity. | Mostly distinct sources. | All sources distinct and from different domains. | The same source cited 3+ times without acknowledgement. | Manual: reviewer counts unique domains. | Duplication is acceptable when a single source genuinely covers a question; the rubric measures UNACKNOWLEDGED duplication. |
| Citation validity | Are the cited URLs / references real and reachable? | Any citation is fabricated or broken. | One or more citations broken. | Most citations valid. | All citations valid and reachable. | A fabricated citation. | Manual: reviewer fetches the citation. | "Reachable" means HTTP 200 on the cited URL at review time. |
| Citation support | Does the cited source actually support the claim? | Citations are present but unrelated to claims. | Some citations support their claim. | Most citations support. | Every cited source supports its claim. | A claim is contradicted by its own citation. | Manual: reviewer reads the cited passage. | This is the most important dimension. |
| Citation completeness | Are all important claims cited? | Multiple important claims have no citation. | Some important claims cited. | Most important claims cited. | All important claims cited. | A number, date, or specific claim is uncited. | Manual: reviewer flags every uncited specific claim. | "Important" is defined by the case's `critical_claims_to_inspect`. |
| Contradiction handling | Are contradictions between sources surfaced? | Contradictions silently merged or hidden. | One contradiction surfaced. | Most contradictions surfaced. | All contradictions surfaced with their sources. | A known contradiction is silently merged. | Manual: reviewer checks `contra-*` cases specifically. | N/A for cases that have no known contradiction. |
| Fact / inference / uncertainty separation | Are facts, inferences, and uncertainties clearly labeled? | Inferences and uncertainties presented as facts. | Some separation, some conflation. | Mostly separated. | All facts, inferences, and uncertainties clearly labeled. | An uncertainty presented as a fact (e.g. "X is 42" when no source confirms). | Manual: reviewer tags every claim. | "Uncertainty" includes both "I do not know" and "the sources disagree". |
| Clarity | Is the report readable and well-structured? | Incoherent or unreadable. | Readable but poorly structured. | Well-structured, minor issues. | Clear, well-structured, easy to follow. | The report is not understandable. | Manual: reviewer reads the report as a user would. | "Clear" is judged on a user who knows the question, not on a domain expert. |
| Cost | Was the per-job cost within the expected class? | Cost is more than 3x the expected class. | Cost is 2-3x the expected class. | Cost is 1-2x the expected class. | Cost is within the expected class. | Cost is more than 5x the expected class. | Recorded in the run manifest. | Cost is recorded, not judged. |
| Latency | Was the per-job latency within the expected class? | Latency is more than 3x the expected class. | Latency is 2-3x the expected class. | Latency is 1-2x the expected class. | Latency is within the expected class. | Latency is more than 5x the expected class. | Recorded in the run manifest. | Latency is recorded, not judged. |
| Stopping quality | Did the service stop work at an appropriate point? | The service went into an obvious loop. | The service did one extra round. | The service stopped at a reasonable point. | The service stopped at the right point. | The service produced a clearly redundant round. | Manual: reviewer counts the rounds. | "Appropriate" is judged by the case, not globally. |
| Sovereignty and egress auditability | Could the owner audit where the report went? | The owner cannot tell which sources were fetched. | The owner can guess from the report. | The owner can identify the sources. | The owner can identify the sources AND the order they were fetched. | The owner cannot tell which sources were fetched. | Manual: reviewer cross-references the report's cited sources to the manifest's `source_urls`. | "Auditability" is not the same as "private". The current pipeline is not designed to keep sources private; it is designed to make the fetch visible. |

No universal threshold is frozen in this draft. The owner will
calibrate thresholds after the pilot. Thresholds such as "E3
score < 10%" would be arbitrary; the calibration is what gives
threshold values meaning.

## 10. Manual audit procedure

The first pilot MUST use structured manual review. The procedure
below is what the human reviewer follows for each of the 16
reports.

For every report:

1. **Identify the important claims.** The reviewer reads the report
   and tags every specific claim (number, date, named entity,
   named work, quoted passage, technical claim). The
   `critical_claims_to_inspect` field of the case sheet is a
   mandatory starting list; the reviewer may add more.
2. **Record the attached citation.** For every important claim,
   the reviewer records the citation exactly as the report states
   it (URL, document name, page, or section).
3. **Check whether the source exists.** The reviewer fetches the
   cited source. If the URL is broken or the document does not
   exist, the claim fails `Citation validity` at 0.
4. **Check whether the cited passage supports the claim.** The
   reviewer reads the cited passage. If the source does not
   support the claim (or contradicts it), the claim fails
   `Citation support` at 0.
5. **Identify important uncited claims.** The reviewer flags
   specific claims with no citation. These fail
   `Citation completeness`.
6. **Record source authority.** For each unique source, the
   reviewer classifies the source against
   `expected_primary_source_types` (regulator, vendor docs, RFC,
   peer-reviewed, etc.).
7. **Detect duplicated or syndicated sources.** The reviewer
   counts unique domains. The same domain cited 3+ times without
   acknowledgement fails `Source diversity and deduplication`.
8. **Record contradictions.** For each known contradiction (in
   the `contra-*` family), the reviewer checks whether the report
   surfaced the contradiction or silently merged.
9. **Separate fact, inference, and uncertainty.** The reviewer
   tags every important claim as fact (with citation), inference
   (with reason), or uncertainty (with source of the uncertainty).
10. **Log reviewer disagreement.** If two reviewers disagree on a
    dimension, the disagreement is recorded in the
    reviewer-disagreement log. Disagreements are NOT resolved by
    averaging; they are surfaced for the owner.

**Aggregation rule (per-claim flags to per-dimension scores).**
Steps 3, 4, 5, 7, and 8 produce per-claim binary flags (the claim
fails dimension X at 0 / the claim does not fail). The rubric in
§9 is per-dimension on a 0–3 gradient. Reviewers aggregate the
per-claim flags to a per-dimension score using the §9 anchors
("most support" / "some support" / "all support"). A single
unsupported important claim does NOT automatically collapse the
dimension to 0; the reviewer chooses 1, 2, or 3 from the anchors
based on the proportion and severity of unsupported claims. The
`Unacceptable failure` column is a HARD VETO that overrides the
gradient: a single `Unacceptable failure` (e.g. fabricated
citation, claim contradicted by its own source) collapses the
dimension to 0 regardless of other claims. The same aggregation
rule applies to `Citation validity`, `Citation completeness`,
`Source diversity and deduplication`, `Contradiction handling`,
and `Fact / inference / uncertainty separation`. When the LLM
draft extraction and the human reviewer disagree, the human
reviewer's verdict is recorded as the rubric value and the LLM
draft is recorded alongside it in the disagreement log.

An LLM MAY assist with steps 1, 2, 5, 6, 7, and 9 as a draft
extraction tool. The LLM's output is NOT gold truth. The LLM's
output is reviewed by the human reviewer. LLM agreement is NOT
treated as independent evidence. Ambiguous decisions remain
owner- or human-adjudicated.

## 11. Pilot outputs (future, NOT in this slice)

When the owner approves the pilot, the pilot will produce:

1. A completed run manifest (16 records).
2. 16 preserved reports (under the owner's chosen location).
3. 8 case scorecards (one per case, with both runs side by side).
4. Claim-and-citation audit sheets (one per report).
5. A reviewer-disagreement log.
6. A variance summary (per dimension, per case).
7. A cost and latency summary (per case, per run).
8. A dominant-failure analysis (or the explicit finding that no
   failure is dominant).
9. A recommendation of exactly one next experiment, or an explicit
   "no change" recommendation.

The pilot outputs are NOT in this slice. This slice only describes
the protocol.

## 12. Decision rules after measurement (future, NOT in this slice)

After the pilot, the dominant observed failure (if any) maps to
exactly one proposed next experiment:

| Dominant failure | Proposed next experiment (NOT yet approved) |
| --- | --- |
| Citation support / completeness | Claim parser and citation verifier experiment. |
| Multi-branch coverage | Static query-decomposition experiment. |
| Source quality / deduplication | Source-policy experiment. |
| Cost or redundant search | Stopping / depth experiment. |
| Contradiction handling | Contradiction experiment. |
| No dominant failure | No pipeline change. |

Each proposed experiment is a future slice, not a current one.
None of them is approved in this slice. None of them is implemented
in this slice. Each will require its own owner authorization.

## 13. Stop and quality gates

The pilot MUST NOT begin until the owner explicitly approves ALL of
the following:

1. The 8 pilot cases (or a revised subset).
2. The rubric draft (or a revised version).
3. The exact baseline commit (`b95afb4943a855eb0cc4fdd911218bbf0d6087b6`
   or a later owner-approved commit).
4. The selected model identifier and provider.
5. The selected search backend.
6. The maximum spend (in USD) for the entire pilot.
7. The execution environment.
8. How the resulting reports are stored and for how long.
9. The reviewer workflow (number of reviewers, conflict resolution).

Until the owner approves all nine, the pilot does not begin, no
provider credentials are requested, and no benchmark is executed.

## 14. Open owner decisions (genuine, not implementation detail)

The following decisions are open. They are NOT implementation
details; each is a real product decision the owner must make.

1. **Pilot case selection.** Accept the 8 cases above as-is, or
   replace one or more with a case the owner prefers. (Cases
   personal, medical, or personalized-financial are NOT in the
   proposal and will NOT be added.)
2. **Pilot corpus version freeze.** Freeze `corpus-v0.1-draft` or
   request a revision. The freeze decides when new cases require
   a new corpus version.
3. **Rubric dimensions.** Accept the 15 dimensions above as-is, or
   add/remove dimensions. Each dimension is a real review load.
4. **Provider selection.** Pick the exact model and provider for
   the pilot. The owner may pick a local-only model, a cloud
   model, or both.
5. **Maximum pilot spend.** Set the cap (in USD) for the entire
   16-job pilot.
6. **Storage of resulting reports.** Decide where the 16 reports
   live, for how long, and who can read them.
7. **Reviewer roster.** Decide who the human reviewers are. One
   reviewer per report is the minimum; two reviewers per report
   is preferred for the dominant-failure analysis.
8. **Decision rule after the pilot.** Accept the 6-row decision
   mapping above as-is, or adjust it. The decision rule governs
   which future experiment (if any) the calibration result maps to.
9. **Future-experiment authorization timing.** Decide whether the
   next experiment is authorized as part of the pilot result, or
   whether each future experiment will require its own
   authorization.

The owner is NOT asked to decide implementation details that
measurement has not yet justified (exact prompts, exact verifier
architecture, exact stopping policy, exact contradiction
detection rules). Those are deferred until a measurement shows
they are warranted.
