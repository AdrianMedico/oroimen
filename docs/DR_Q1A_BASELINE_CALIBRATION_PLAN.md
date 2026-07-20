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

- **Family assignment must be defensible.** Every case is
  assigned to exactly one of the 8 families. The assignment
  must be defensible at freeze time: the case's primary
  failure-mode risk must be the defining risk of the assigned
  family, and the reviewer must be able to use the family
  context to choose the rubric anchors. A multi-risk case
  (for example, a question that is both "recent factual" and
  "contradictory sources") is assigned to the family whose
  defining risk is the strongest signal, and the secondary risk
  is recorded in `owner_notes` so the reviewer can apply the
  family-specific vetoes of both.
- **Scope is bounded.** Every case specifies the entities
  (airline names, country names, product SKUs, etc.) by name, the
  geography, the date range, the cabin or fare class, the route
  or route class, and any other dimension that constrains the
  answer. A case that requires a ranking basis (for example,
  "the three most-cited papers") must define the bibliographic
  database, the publication type, the date range, the selection
  rule, and the tie-break rule.
- **Auditability is practical.** Every case is auditable from
  primary sources the reviewer can fetch independently. A case
  whose primary sources are behind authentication, paywalls, or
  private databases is rejected.
- **No false premise is silently embedded** unless detecting the
  false premise is explicitly the intended test. A case whose
  question embeds a false premise (for example, "why does X
  always Y when X is a known myth?") is acceptable IF AND ONLY
  IF the case sheet's `unacceptable_failures` list explicitly
  includes "report treats the premise as true". Otherwise the
  case is rewritten.
- **Freshness metadata is meaningful.** Every case has
  `evaluated_at` and `stale_after`. Volatile cases (the recent
  factual and travel and logistics families) have a non-trivial
  `stale_after` that the reviewer can use to flag the case at
  run time. Non-volatile cases have `stale_after = null` or a
  date far in the future, recorded in the case sheet.
- **No unverifiable ranking.** A case that requires a single
  authoritative ranking (for example, "the top three", "the
  best", "the largest") must define the ranking basis AND a
  tie-break rule. Otherwise the case is rewritten.
- **Source expectations are clear.** Every case's
  `expected_primary_source_types` is non-empty and matches the
  family. A case that does not specify expected source types
  is incomplete and is rejected at freeze time.
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
| rec-01 | Recent factual | Select three representative peer-reviewed benchmark papers on long-context LLM evaluation. Use Semantic Scholar (semanticscholar.org) as the bibliographic database, restrict to publication type "Journal" or "Conference" with publication date in calendar year 2025, and choose the three papers with the highest Semantic Scholar `citationCount` for the query "long-context LLM evaluation". If citation counts are equal, prefer the paper with the more recent publication date; if still tied, pick the one with the higher Semantic Scholar `influentialCitationCount`; document any tie-break rule applied. | stale_after=2026-12-31 |
| rec-02 | Recent factual | As of evaluation date 2026-07-20, what is the documented state of EU AI Act enforcement actions against foundation-model providers? Use only official EU sources: the European AI Office (digital-strategy.ec.europa.eu/policies/ai-office), the Official Journal of the EU (eur-lex.europa.eu), and the European Commission's AI Act pages. Record the source URL and the published date for every enforcement claim. | stale_after=2026-12-31 |
| rec-03 | Recent factual | Enumerate CVE-class RCE vulnerabilities disclosed in the NVD (nvd.nist.gov) and the project-specific advisories for OpenWrt (openwrt.org) and pfSense (pfsense.org) in the 12-month window [2025-07-20, 2026-07-20]. For each CVE, record: CVE id, CVSS v3.1 base score >= 7.0, affected product and version range, disclosure date, and the primary advisory URL. "RCE" means CVSS v3.1 vector contains "C" (Confidentiality), "I" (Integrity), and "A" (Availability) all High, with attack vector Network and attack complexity Low, OR an explicit "Remote Code Execution" tag in the CVE description. | stale_after=2026-12-31 |
| arch-01 | Technical architecture | Compare the architectures of PostgreSQL's MVCC implementation and FoundationDB's record-layer implementation. | low |
| arch-02 | Technical architecture | Explain how Rust's borrow checker handles async closures. Cite the relevant language reference sections. | low |
| arch-03 | Technical architecture | How does the Linux kernel's cgroup v2 freezer interact with systemd-managed services? | low |
| prod-01 | Product comparison | Compare 1Password Teams, Bitwarden Business, and Passbolt Starter across feature sets, pricing models, and self-hosting options, as of 2026-07-20. | moderate |
| prod-02 | Product comparison | Compare the OCR engines Tesseract, PaddleOCR, and Surya for historical document transcription. | low |
| prod-03 | Product comparison | Compare the local-first note-taking apps Obsidian, Logseq, and Anytype across data ownership, sync, and plugin models. | low |
| reg-01 | Regulation | Summarize the data-residency requirements for clinical data under HIPAA and the GDPR. | low |
| reg-02 | Regulation | What is the current process for filing a security advisory with the Python Security Response Team? | moderate |
| reg-03 | Regulation | Summarize the European Accessibility Act requirements for e-commerce sites, with primary sources. | low |
| travel-01 | Travel and logistics | Compare the standard checked-baggage allowance and weight limits for British Airways (oneworld), Delta Air Lines (SkyTeam), and United Airlines (Star Alliance) on a single transatlantic economy-class round-trip fare booked 2026-07-20, departing LHR-JFK 2026-09-15 and returning JFK-LHR 2026-09-22, fare family "Economy (Light)" or its airline-specific equivalent for each carrier, evaluated as of 2026-07-20. | stale_after=2026-12-31 |
| travel-02 | Travel and logistics | As of 2026-07-20, what is the standard entry policy (visa-free / visa-on-arrival / e-visa, whichever applies) for a German passport holder entering each of Indonesia, Thailand, the Philippines, and Vietnam? Rank the four countries by 2024 international tourist arrivals from the UNWTO Compendium of Tourism Statistics, dataset "Tourism arrivals by region of origin". Tie-break: if two countries have equal 2024 arrivals, prefer the country with the longer standard maximum stay in days for a German passport holder; if still tied, alphabetical order by country name. Document any tie-break rule applied. Record the entry requirements, the maximum stay in days, and the official immigration-authority URL for each country. | stale_after=2026-12-31 |
| travel-03 | Travel and logistics | Summarize the current pet-import requirements for cats and dogs entering the UK from the EU. | stale_after=2026-12-31 |
| contra-01 | Contradictory sources | Summarize the current evidence on the effect of intermittent fasting on insulin resistance, citing primary sources that disagree. | moderate |
| contra-02 | Contradictory sources | What does the evidence say about coffee consumption and cardiovascular risk? Surface the disagreement. | moderate |
| contra-03 | Contradictory sources | How effective is static typing at preventing bugs in large codebases? Surface both sides. | moderate |
| multi-01 | Multi-branch | Plan a 7-day trip to Japan from 2026-10-24 to 2026-10-30 inclusive for a vegetarian family of four (two adults, two children aged 8 and 12) on a moderate budget of approximately EUR 6,000 total excluding international flights, arriving at Tokyo Narita (NRT) and departing from Osaka Kansai (KIX). Required destinations in order: Tokyo (3 nights), Hakone (1 night), Kyoto (2 nights), Nara day-trip from Kyoto, Osaka (1 night). Compare the Japan Rail Pass (JR Pass) 7-day ordinary adult price against a calculated point-to-point Shinkansen + local train itinerary that covers Tokyo-Hakone-Kyoto-Nara-Osaka; record both totals in JPY and EUR at the exchange rate published by the European Central Bank on 2026-07-20. | low |
| multi-02 | Multi-branch | Summarize the past 5 years of the Python packaging story (PEP 517, PEP 518, PEP 621, pyproject.toml, uv, hatchling). | low |
| multi-03 | Multi-branch | Compare the data-protection regimes of Brazil (LGPD), California (CCPA/CPRA), and the EU (GDPR) for a small SaaS company. | low |
| uncert-01 | Legitimate uncertainty | Predict the most likely 2027 standardization outcome for the W3C Web Neural Network API (WebNN). State the prediction, the probability assigned to each plausible outcome (W3C Recommendation, W3C Working Draft, deprecation, no consensus), the W3C working group status as of evaluation date 2026-07-20, and at least two independent evidence sources for the probability assignment. The expected disposition is an explicit "unknown" with reasoned scenario analysis, NOT a single point forecast. | explicit "unknown" disposition required |
| uncert-02 | Legitimate uncertainty | Forecast the average spot price in USD of one Corsair Vengeance 32GB (2x16GB) DDR5-6000 CL30 desktop memory kit (CMK32GX5M2B6000C30) on newegg.com (US) on 2026-12-31. Use the historical price series from camelcamelcamel.com (or pcpartpicker's price history) for the trailing 365 days as the evidence window. State the forecast, the confidence interval, the model class used (e.g. random-walk, ARIMA, naive seasonal), and at least two external supply-chain signals (e.g. TrendForce DRAM contract-price reports, manufacturer guidance). The expected disposition is an explicit forecast with uncertainty band, NOT a single point prediction. | explicit "unknown" disposition required |
| uncert-03 | Legitimate uncertainty | A residential customer in Toronto, Ontario wants to know whether the average peak household demand on a typical weekday in January 2027 is more likely to be above or below 4.5 kW, given the 2023-2026 trend in Toronto Hydro published distribution-system data and Statistics Canada household energy-use statistics. State the prediction, the confidence band, the evidence sources, and the assumptions (e.g. electric-vehicle charging excluded, gas-heated home). The expected disposition is a reasoned scenario analysis with a stated "unknown" range, NOT a single point forecast. | explicit "unknown" disposition required |

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

**Per-family statistical limit (explicit).** One case per family
supports a per-case conclusion only. It does NOT support a
conclusion that a whole family is good or bad, and it does NOT
support a conclusion that a failure observed in one family is
absent in another. With n=1 per family, the per-family variance
is only meaningfully detectable as "fully stable" or "fully
inverted" between the two runs; finer variance is unresolvable.
The pilot's role is to detect strong candidate failure modes and
to confirm the rubric and the review workflow are usable, NOT to
generalize to the family or to the corpus.

The pilot explicitly excludes any personal, medical, or
personalized-financial case. No provider spending happens in this
slice.

### 6.1 Expansion gate to the remaining 16 cases

The 24-case corpus is the eventual measurement; the 8-case
pilot is a variance and rubric-validity pilot. The remaining
16 cases (the 2 non-pilot cases per family) are NOT executed
automatically after the 8-case pilot. They are gated by the
following 6 conditions; ALL six must be true before any of the
remaining 16 cases is executed.

1. **Rubric usable.** Two reviewers applying the rubric
   independently to the same report agree on every graded
   dimension within ±1 point at least 80% of the time. Below
   80%, the rubric itself is defective and the pilot halts.
2. **Review time acceptable.** The average per-report review
   time across the 8-case pilot is within the owner-approved
   reviewer-time budget.
3. **Cases auditable.** At least 7 of the 8 pilot cases were
   auditable end-to-end (the reviewer could fetch the primary
   sources and verify the claims). A case that cannot be
   audited end-to-end is a corpus defect, not a pipeline
   defect; it is rejected and the corpus is revised.
4. **Disagreement manageable.** The reviewer-disagreement log
   for the 8-case pilot has fewer than 3 unresolved dimension
   disagreements per case on average. More than that means the
   rubric is too ambiguous for the corpus.
5. **Cost within owner cap.** Total spend for the 8-case pilot
   is at or below the owner-approved cap. The remaining 16
   cases receive a separate cap.
6. **No critical protocol defect.** No P0/P1 finding was
   identified in the protocol itself (e.g. a manifest field
   that was not actually recorded, a rubric dimension that
   cannot be applied to any pilot case, a privacy leak in the
   report path). A critical protocol defect halts the
   measurement and the protocol is revised.

If any of the six conditions is not met, the pilot is NOT
expanded; the 8-case result is published, the owner decides
whether to revise the protocol, revise the corpus, expand the
spend cap, or halt the calibration entirely. The remaining
16 cases are NOT authorized by the 8-case pilot.

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
| `total_latency_s` | Wall-clock duration (recorded, NOT judged against a class). |
| `tokens_in` | Sum of input tokens across all LLM calls. |
| `tokens_out` | Sum of output tokens across all LLM calls. |
| `cost_usd` | Recorded per-job cost (recorded, NOT judged against a class). |
| `final_status` | `complete` / `failed` / `cancelled`. |
| `terminal_status` | Same value as `final_status`; the pipeline has only one terminal state. |
| `truncation_observed` | `true` if any `llm.chat` call was truncated by `max_tokens`; `false` otherwise. |
| `repetition_observed` | `true` if the per-source synthesis produced content that was later repeated verbatim in the final synthesis. |
| `retry_count` | Integer count of retry or recovery events during the run. |
| `wasted_work_observed` | `true` if any fetched source was NOT used in the final synthesis. |
| `source_urls` | The list of source URLs the service actually fetched. |
| `source_urls_in_fetch_order` | The same list in the order the fetcher actually called them (not the order they appear in the report). |
| `report_artifact_ref` | A reference (NOT a path) to the report artifact. |
| `report_storage_destination` | The location (NOT a filesystem path) where the report artifact is stored. |
| `search_provider_egress` | Search provider call records: query text, response count, provider name, timestamp. |
| `llm_provider_egress` | LLM provider call records: phase, tokens in/out, model, timestamp. |
| `notifier_egress` | Notifier call records: notifier name, function called, arguments (no message body). |
| `retry_information` | Any retry or recovery events. |
| `reviewer` | The handle of the human reviewer. |
| `rubric_version` | The rubric version used. |

No credentials, no API keys, no raw filesystem paths, and no
internal exception text appear in the manifest. Sensitive values
stay under the owner's chosen storage. The cost and latency
fields are recorded as raw numbers, never as classes. The
egress sub-fields are recorded as a list of records with the
fields above.

## 9. Draft quality rubric

The draft rubric has **15 dimension-level entries** below, with
one of them (Sovereignty and egress auditability) scored as 5
sub-dimensions. Each dimension-level entry uses one of three
modes:

- **Graded (0-3):** 12 dimension-level entries, with explicit
  anchors per score, an Unacceptable-failure veto, and a
  deterministic aggregation rule (§9.1). Of these, 11 are
  individual dimensions and 1 is the Sovereignty dimension whose
  score is the 5-tuple of its sub-scores.
- **Descriptive:** 2 dimensions (Cost, Latency), recorded as raw
  numbers in the manifest. No 0-3 score. No pre-pilot class.
  Empirical distribution is the only comparison.
- **N/A for current baseline:** 1 dimension (Stopping), recorded
  as raw descriptive fields in the manifest. No 0-3 score for
  the current pipeline. Future iterations of the pipeline may
  score this.

No universal threshold is frozen in this draft. The owner will
calibrate thresholds after the pilot. Thresholds such as "E3
score < 10%" would be arbitrary; the calibration is what gives
threshold values meaning. A `1` is **not** a passing score; a
`3` is the only clean-pass score; `2` is acceptable with minor
issues; `0` is a regression that the reviewer must describe.

### 9.0 Dimensions

| Dimension | Mode | What is judged | 0 anchor | 1 anchor | 2 anchor | 3 anchor | Unacceptable failure (veto) | Evidence | Ambiguity notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Factual accuracy | Graded | Consistency with the best available evidence after considering all relevant sources. | Multiple false claims. | Some correct, some unverifiable. | Mostly correct, minor inaccuracies. | All verifiable claims correct. | A claim asserted as fact is contradicted by the best available evidence. | Manual; reviewer considers all relevant sources, not only the cited ones. | "Best available evidence" includes the cited source AND any related sources the reviewer can locate. |
| Citation support | Graded | Whether the cited source actually supports the attached claim. | Citations present but unrelated to claims. | Some citations support their claim. | Most citations support. | Every cited source supports its claim. | A claim is directly contradicted by its own citation. | Manual; reviewer reads the cited passage. | A cited source that does not address the specific claim but is in a related field scores 0 unless the report acknowledges the gap. |
| Citation validity | Graded | The reference identity is valid (real DOI, real URL, real document). | Multiple citations have non-existent reference identity. | One or more citations have invalid identity. | Most citations have valid identity. | All citations have valid reference identity. | A citation with a fabricated reference identity (a made-up DOI, a URL that never existed). | Manual; reviewer checks DOI resolution, URL existence via archive.org if the live URL is unavailable. | An invalid identity is NOT the same as an unavailable source (see Source availability). |
| Source availability | Graded | The source was retrievable, the content was available, and support could be evaluated. | Multiple sources were not retrievable. | One or more sources were not retrievable. | Most sources were retrievable. | All sources were retrievable. | (No veto; this is a descriptive dimension, not a quality veto.) | Recorded in the manifest. | A 200 OK from a landing page is sufficient; the specific passage need not be retrievable if the URL is real. Redirects, DOI landing pages, PDFs, bot blocks, authentication walls, and post-run availability changes do NOT automatically become fabricated citations. |
| Citation completeness | Graded | All important claims are cited. | Multiple important claims have no citation. | Some important claims cited. | Most important claims cited. | All important claims cited. | A claim listed in the case's `critical_claims_to_inspect` has no citation. | Manual; reviewer checks each claim against the citation list. | "Important" is defined by the case sheet. |
| Recency | Graded | Sources are within the case's required evidence window. | All sources are outside the required window. | Some sources are within the window. | Most sources are within the window. | All sources are within the required window. | A case marked with a `freshness_level` other than `low` produces a report whose primary sources are all older than `case.stale_after` AND the report does not acknowledge staleness. | Manual; reviewer checks each source's publication or last-update date. | `case.stale_after` is the EXPIRATION of the case sheet; it bounds the run, not the source. Source recency is judged relative to (a) the case's `evaluated_at`, (b) the task's required evidence window, and (c) each source's own publication or last-update date. |
| Source authority | Graded | Sources are authoritative for the question. | Sources are low-authority for the question. | Mixed authority. | Mostly authoritative. | Sources are the canonical authority for the question. | A primary source listed in the case's `expected_primary_source_types` is missing AND no acknowledgement. | Manual; reviewer compares to `expected_primary_source_types`. | "Authoritative" is judged per-family, not globally. |
| Evidence independence | Graded | Sources are editorially and evidentially independent. | Multiple sources are duplicates, syndication, or share the same underlying evidence. | Some duplication, some independent sources. | Mostly independent. | All sources are editorially and evidentially independent. | 3+ citations resolve to the same underlying document without acknowledgement. | Manual; reviewer checks domain, editorial ownership, and shared content. | Domain count MAY be recorded in the manifest but is not the sole criterion; multiple pages from one canonical authority (e.g. EU regulation chapters) are legitimate. |
| Contradiction handling | Graded | Contradictions between sources are surfaced. | Contradictions silently merged or hidden. | One contradiction surfaced. | Most contradictions surfaced. | All contradictions surfaced with their sources. | A known contradiction (in the `contra-*` family) is silently merged. | Manual. | N/A for cases that have no known contradiction; reviewer marks N/A. |
| Fact / inference / uncertainty separation | Graded | Facts, inferences, and uncertainties are clearly labeled. | Inferences and uncertainties presented as facts. | Some separation, some conflation. | Mostly separated. | All facts, inferences, and uncertainties clearly labeled. | An uncertainty presented as a fact (e.g. "X is 42" when no source confirms). | Manual; reviewer tags every claim. | "Uncertainty" includes both "I do not know" and "the sources disagree". |
| Clarity | Graded | Report is readable and well-structured. | Incoherent or unreadable. | Readable but poorly structured. | Well-structured, minor issues. | Clear, well-structured, easy to follow. | The report is not understandable. | Manual; reviewer reads the report as a user would. | "Clear" is judged on a user who knows the question, not on a domain expert. |
| Completeness | Graded | The answer covers the expected subquestions. | Major branches missing. | One or more branches partially covered. | Most branches covered. | All expected branches covered. | A whole branch missing AND no acknowledgement. | Manual; reviewer checks the `expected_subquestions`. | For multi-branch cases, "covered" means at least one citation per branch. |
| Cost | Descriptive | Per-job cost. | n/a | n/a | n/a | n/a | n/a | Recorded in the manifest as `cost_usd`. | No class, no threshold, no comparison before the pilot. |
| Latency | Descriptive | Per-job latency. | n/a | n/a | n/a | n/a | n/a | Recorded in the manifest as `total_latency_s`. | No class, no threshold, no comparison before the pilot. |
| Stopping | N/A for baseline | Whether the service stopped at a sensible point. | n/a | n/a | n/a | n/a | n/a | Recorded in the manifest as `truncation_observed`, `repetition_observed`, `retry_count`, `wasted_work_observed`. | The current single-pass pipeline does not have adaptive stopping; the 0-3 anchors for adaptive stopping are reserved for future iterations of the pipeline. |
| Sovereignty and egress auditability (5 sub-dimensions) | Graded per sub-dimension | See §9.2. | See §9.2. | See §9.2. | See §9.2. | See §9.2. | See §9.2. | Recorded in the manifest. | If the current runtime cannot expose one of these, the score is recorded as UNKNOWN or UNMEASURED, NOT as 0. |

**Total graded dimension-level entries: 12 (11 individual + 1
Sovereignty dimension whose 5 sub-dimensions are scored
separately and NOT averaged).** The 11 individual graded
dimensions are scored 0-3 with the anchors above. The
Sovereignty dimension is recorded as the 5-tuple of its
sub-scores (§9.2). The 2 descriptive dimensions are recorded as
raw numbers. The 1 N/A-for-baseline dimension (Stopping) is
recorded as raw descriptive fields.

### 9.1 Aggregation rule (deterministic per-claim to per-dimension mapping)

Steps 3-10 of the §10 procedure produce per-claim binary flags
(the claim fails dimension X at 0 / the claim does not fail).
The 11 individual graded dimensions are scored 0-3 on a per-case
basis using the provisional percentage bands below. The bands
are explicitly marked for pilot calibration; the owner will
adjust them after the pilot, not before.

For each graded dimension, the reviewer counts the
proportion of inspected claims (from `critical_claims_to_inspect`
plus the reviewer's own additions) that the dimension finds
"compliant":

| Dimension score | Provisional band (compliance fraction) | Anchor meaning |
| --- | --- | --- |
| 0 | < 50% of inspected claims compliant | Anchor 0. |
| 1 | 50% to < 70% | Anchor 1. |
| 2 | 70% to < 90% | Anchor 2. |
| 3 | >= 90% | Anchor 3. |

**Hard veto.** The `Unacceptable failure` column above is a HARD
VETO that overrides the gradient: a single veto event collapses
the dimension to 0 regardless of the compliance fraction.
Vetoes are restricted to critical-claim failures — the
`critical_claims_to_inspect` and `unacceptable_failures` fields
of the case sheet. A minor uncited number outside the
`critical_claims_to_inspect` list does NOT trigger a veto; it
is recorded as a citation-completeness score reduction but does
not collapse the dimension to 0.

**Inter-rater agreement.** Two reviewers applying the rule
independently to the same report should agree on the
dimension score within ±1 point at least 80% of the time
across the pilot. Below 80%, the rubric itself is defective
and the pilot halts (§13 decision gate).

### 9.2 Sovereignty and egress auditability (5 sub-dimensions)

The single `Sovereignty and egress auditability` dimension is
split into 5 sub-dimensions, each scored 0-3 with the same
anchor pattern. The sub-dimensions correspond to distinct egress
channels. If the current runtime cannot expose one of these,
the score is recorded as UNKNOWN or UNMEASURED, NOT as 0.

| Sub-dimension | What is judged | 0 anchor | 1 anchor | 2 anchor | 3 anchor | UNKNOWN / UNMEASURED |
| --- | --- | --- | --- | --- | --- | --- |
| `search_provider_egress` | The owner can identify which search provider was called, the query sent, and the result count. | Cannot tell. | Can guess. | Can identify. | Can identify AND the timestamp and the URL of the result page. | If the runtime does not surface this field, the score is UNKNOWN. |
| `fetched_source_urls` | The owner can identify which URLs were fetched, in fetch order. | Cannot tell. | Can guess from the report. | Can identify the URLs. | Can identify the URLs AND the order they were fetched (`source_urls_in_fetch_order`). | If the runtime does not surface fetch order, the score for the "order" component is UNMEASURED. |
| `llm_provider_egress` | The owner can identify which LLM provider was called, how many times, and how many tokens per call. | Cannot tell. | Can guess. | Can identify. | Can identify AND the phase, the tokens in/out, and the timestamp per call. | If the runtime does not surface phase or timestamp, those components are UNMEASURED. |
| `notifier_egress` | The owner can identify which notifier was called, with which function, and with which argument list (no message body). | Cannot tell. | Can guess. | Can identify. | Can identify AND the timestamp and the argument list. | If the runtime does not surface the argument list, that component is UNMEASURED. |
| `report_storage_destination` | The owner can identify where the final report is stored. | Cannot tell. | Can guess. | Can identify the storage location. | Can identify the storage location AND the retention policy. | If the runtime does not surface the retention policy, that component is UNMEASURED. |

The 5 sub-scores are recorded separately in the manifest and
are NOT averaged into a single 0-3 score for the dimension.
The dimension-level report for Sovereignty and egress
auditability is the 5-tuple, not an average.

## 10. Manual audit procedure

The first pilot MUST use structured manual review. The procedure
below is what the human reviewer follows for each of the 16
reports. The procedure maps step-by-step to the 12 graded
dimensions and the 5 sovereignty/egress sub-dimensions in §9.

For every report:

1. **Identify the important claims.** The reviewer reads the
   report and tags every specific claim (number, date, named
   entity, named work, quoted passage, technical claim). The
   `critical_claims_to_inspect` field of the case sheet is a
   mandatory starting list; the reviewer may add more. The
   total inspected-claim count is recorded.
2. **Record the attached citation.** For every important claim,
   the reviewer records the citation exactly as the report
   states it (URL, document name, page, or section).
3. **Check the reference identity (Citation validity).** The
   reviewer resolves the DOI (if present) or fetches the URL
   (with a documented retry on a 5xx response). If the identity
   does not resolve and the URL has no archive.org snapshot, the
   claim's reference identity is invalid. A 200 OK from a landing
   page is sufficient for the identity; the specific passage
   need not be retrievable. Redirects, DOI landing pages, PDFs,
   bot blocks, authentication walls, and post-run availability
   changes do NOT automatically become fabricated citations.
4. **Check whether the cited passage supports the claim
   (Citation support).** The reviewer reads the cited passage
   (or the landing page if the passage is not retrievable). If
   the source does not support the claim (or contradicts it),
   the claim is unsupported.
5. **Check source availability (Source availability).** The
   reviewer records whether the source was retrievable at
   review time, separately from the validity check. The two
   dimensions are independent: a real URL may be temporarily
   down, a broken URL may be valid historically.
6. **Identify important uncited claims (Citation completeness).**
   The reviewer flags specific claims with no citation. A
   claim listed in `critical_claims_to_inspect` with no
   citation triggers the Unacceptable-failure veto.
7. **Record source authority.** For each unique source, the
   reviewer classifies the source against
   `expected_primary_source_types` (regulator, vendor docs,
   RFC, peer-reviewed, etc.).
8. **Check evidence independence.** The reviewer records the
   domain AND the editorial ownership AND any shared content
   between sources. The same underlying document cited 3+
   times without acknowledgement triggers the veto. Multiple
   pages from one canonical authority (e.g. EU regulation
   chapters) are legitimate and are NOT counted as duplication.
   Domain count is recorded in the manifest but is NOT the
   sole criterion.
9. **Record contradictions.** For each known contradiction (in
   the `contra-*` family), the reviewer checks whether the
   report surfaced the contradiction or silently merged.
10. **Separate fact, inference, and uncertainty.** The reviewer
    tags every important claim as fact (with citation),
    inference (with reason), or uncertainty (with source of the
    uncertainty).
11. **Check completeness.** The reviewer walks the
    `expected_subquestions` list and marks each as covered,
    partial, or missing.
12. **Check clarity.** The reviewer reads the report as a user
    would and records the clarity score.
13. **Record the 5 sovereignty/egress sub-dimensions.** The
    reviewer cross-references the manifest's
    `source_urls_in_fetch_order`, `search_provider_egress`,
    `llm_provider_egress`, `notifier_egress`, and
    `report_storage_destination` fields. Each sub-dimension is
    scored 0-3 per §9.2, or marked UNKNOWN/UNMEASURED.
14. **Log reviewer disagreement.** If two reviewers disagree on
    a dimension, the disagreement is recorded in the
    reviewer-disagreement log. Disagreements are NOT resolved
    by averaging; they are surfaced for the owner.

**Aggregation rule.** The aggregation rule is defined in §9.1
and is the single source of truth. The reviewer applies the
provisional percentage bands in §9.1 to the per-claim flags
from steps 3-10. The `Unacceptable failure` column is a HARD
VETO restricted to `critical_claims_to_inspect` failures. A
minor uncited number outside the `critical_claims_to_inspect`
list is recorded as a score reduction but does NOT trigger a
veto. The sovereignty/egress sub-scores are recorded separately
and are NOT averaged into a single dimension score.

**LLM assist.** An LLM MAY assist with steps 1, 2, 3 (URL
existence only), 4 (claim-citation pairing), 5 (HTTP HEAD probe
only), 6, 7, 8 (domain extraction only), 10, and 13
(manifest field reading only) as a draft extraction tool. The
LLM's output is NOT gold truth. The LLM's output is reviewed by
the human reviewer. LLM agreement is NOT treated as
independent evidence. When the LLM draft extraction and the
human reviewer disagree, the human reviewer's verdict is
recorded as the rubric value and the LLM draft is recorded
alongside it in the disagreement log.

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
exactly one proposed next experiment. The phrasing matters: a
finding from the 8-case pilot is a finding about the 8 cases
sampled, NOT a finding about the family, the corpus, or the
pipeline in general.

| Pilot finding | Phrasing used in the pilot report | Proposed next experiment (NOT yet approved) |
| --- | --- | --- |
| Citation support / completeness failure observed in 1+ pilot cases | "Citation support / completeness failure observed in the pilot" | Claim parser and citation verifier experiment (next experiment, NOT yet approved). |
| Multi-branch coverage failure observed in 1+ pilot cases | "Multi-branch coverage failure observed in the pilot" | Static query-decomposition experiment (NOT yet approved). |
| Source quality / evidence-independence failure observed in 1+ pilot cases | "Source quality / evidence-independence failure observed in the pilot" | Source-policy experiment (NOT yet approved). |
| Cost or redundant-search failure observed in 1+ pilot cases | "Cost or redundant-search failure observed in the pilot" | Stopping / depth experiment (NOT yet approved). |
| Contradiction handling failure observed in 1+ pilot cases | "Contradiction handling failure observed in the pilot" | Contradiction experiment (NOT yet approved). |
| No dominant failure observed across the 8 pilot cases | "No dominant failure observed in the pilot" (NOT "no dominant failure exists") | No pipeline change. The owner retains the right to commission a follow-up measurement on the remaining 16 cases before authorising a no-change posture. |

Each proposed experiment is a future slice, not a current one.
None of them is approved in this slice. None of them is implemented
in this slice. Each will require its own owner authorization.

The "no change" outcome is a finding about the 8 cases sampled
with 2 runs each. It is NOT a finding about the family, the
corpus, the pipeline, or the production readiness of the
system. The pilot does not produce a product-readiness verdict.

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

**Expansion gate.** A separate set of six conditions (rubric
usable, review time acceptable, cases auditable, disagreement
manageable, cost within owner cap, no critical protocol defect)
gates the expansion from the 8-case pilot to the remaining
16 cases. See §6.1. The expansion gate is a quality gate, not
an owner approval: the conditions are evaluated objectively
from the pilot output. The owner is informed of the expansion
or the halt, but the owner does not need to re-authorize the
remaining 16 cases unless the conditions fail.

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
3. **Rubric dimensions.** Accept the 15 dimension-level entries
   above (or the 12 graded + 2 descriptive + 1 N/A, by mode) as-is,
   or add/remove dimensions. The Sovereignty dimension has 5
   sub-dimensions; the 5 sub-scores are not averaged. Each
   dimension-level entry is a real review load.
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
