# Failure taxonomy

This directory defines the closed labels for failed tool calls in reproduction run ledgers. It is for ledger labelers and the scripts that read their labels.

## Table of contents

- [Files](#files)
- [Label a recorded failure](#label-a-recorded-failure)
- [Count reproduction errors](#count-reproduction-errors)
- [Label a Scientific contract requirement](#label-a-scientific-contract-requirement)
- [Add a root cause](#add-a-root-cause)

## Files

| File | Holds |
| --- | --- |
| `contract_categories.csv` | The ten Figure 7 contract categories, definitions, and one-category decision rules. |
| `causes.csv` | The four ordered cause tests: Harness, Environment, Source, and Agent. |
| `activities.csv` | The four activities and whether each counts. |
| `root_causes.csv` | The closed root-cause list used for counting, with a cause, whether it counts, a decision test, and a data example for each value. |
| `scientific_requirements.csv` | Masked scientific requirements, their reference values and sources, what hid them, and retained evidence. |
| `scientific_settlement.csv` | How a scientific requirement was settled and whether that settlement fulfills it. |
| `rubric_ids.csv` | MLCommons ontology criteria and categories. |

## Label a recorded failure

0. Decide whether to record the failure. Record a background job or earlier command once: use the poll that returned its nonzero status as the ordinal, otherwise use the call that launched it, and, if neither exists, use the first same-session call that reveals it. Use the call holding the decisive error as `quote_ordinal`. Do not record a later call that only displays an already-recorded failure. Do not record a call the agent interrupted with Ctrl-C, a poll of a job still running or already finished, or a probe whose negative result was intended. Source code that only contains an assertion template, a prior-session log in prior artifacts, and a redirection that only suppresses output are not session failures. Every recorded failure needs a decisive evidence line: the exception or error message from the call, a later same-session log read, or, for a silent exit, the line changed to fix it. A bare exit status or bare `No such file or directory` is not decisive.
1. Apply `causes.csv` in its order. Test Harness first, then Environment, then Source. If none applies, the cause is Agent.
2. Choose the activity in `activities.csv`. Reproduction is the deliverable `reproduce.sh`, the reference steps it runs, and bounded preflight runs of that script. A separate checker the agent wrote is Checking. Probes, reads, searches, environment queries, and skill-file reads are Inspection. Plans, reports, and evidence or profiling bookkeeping are Reporting.
3. Choose exactly one category from `contract_categories.csv`. Use the failed operation and its corrective action. For a bug in the agent's patch or script, use the category of the Implementation item that patch or script was adapting.
4. Choose exactly one `root_cause` from `root_causes.csv`. It is the reusable, plain-domain cause shown by the evidence, not a new wording of the event description.
5. Mark `fixed` true only when a later call in the same session completes the failed operation with the Scientific contract preserved. Bypassing an Implementation-only check, such as a pickle byte checksum that depends on serialization, is fixed when scientific content is preserved.

## Count reproduction errors

The counting unit is one distinct `(session, root_cause)` pair. Count it only when `cause` is Agent, `activity` is Reproduction, and the root cause has `counted` = yes. Retries, repeated manifestations, and a later repair do not add another count. The same root cause in another session counts again.

A root cause has `counted` = no when it cannot change the reproduced content: the failed check compares serialized bytes, such as a file checksum, and the content those bytes encode matches the reference. Such a failure is recorded but not counted.

## Label a Scientific contract requirement

Use `scientific_requirements.csv` to identify each masked requirement. A masked item can contain more than one requirement. Record the settlement from `scientific_settlement.csv`: authoritative evidence still in supplied sources and a human decision are fulfilled; an assumption and an unresolved choice are failures; prior artifacts take the outcome of the basis they record. A matching score does not turn an assumed or unresolved scientific requirement into a fulfilled one.

Set `disclosed` to true only when the session's report labels the value as assumed; mentioning or using the value elsewhere is not disclosure. Set `matches_reference` to whether the agent's value or rule equals the requirement's `reference_value`, independently of how the requirement was settled.

## Add a root cause

When no existing root cause fits, add one row to `root_causes.csv` before labeling the event. Give it a descriptive kebab-case name in plain domain terms, a cause that exists in `causes.csv`, a narrow decision test that distinguishes it from existing rows, and an entry id with a verbatim decisive quote. Never add a free-text event label.
