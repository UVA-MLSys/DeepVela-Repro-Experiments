# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` has three ordered stages: build a Python 3.11 environment
and the repository LoadGen package; preprocess the SQuAD v1.1 validation set;
then run one Offline PyTorch FP32 accuracy pass and the repository evaluator.
The immutable dataset has 10,570 questions and preprocessing produces 10,833
features. Input SHA-256 values are `95aa6a52...72c9` for `dev-v1.1.json`,
`71af14ac...397f` for `model.pytorch`, and `07eced37...038a3` for `vocab.txt`.

## Measured costs and resources

- Clean environment and LoadGen build: 47 seconds.
- Full CPU preprocessing: 60 seconds and 10,833 features.
- 8-feature GPU run plus repository evaluation: 26 seconds.
- Immediately before preflight, the local A100 80GB had 79,323 MiB free.
- The matching provenance-tracked prior campaign reports a clean full duration
  of 20 minutes 55 seconds and a steady observed rate near 12--13 samples/s.

## Schedule and projection

Run `./reproduce.sh --target bert-squad --from setup` with no sample limit, so
the first full run exercises the delivered script from nothing built. Build and
preprocessing use CPU first; the sole GPU run then runs alone, followed by CPU
evaluation. There is no other target whose CPU work can overlap this GPU stage.

The comparable measured duration is 20:55. Allowing for shared-GPU variation,
the projection is 25 minutes. Reserve 10 minutes afterward for output checks
and `report.json`, well within the two-hour wall-clock limit.

## Re-plan triggers

After the first few percent of accuracy records are written, compare the
observed rate with the 25-minute projection. If shared-GPU contention pushes
the projection beyond the limit less the reserve, record that evidence here.
On failure, repair through `--from` so the script clears its own downstream
markers and outputs; never remove benchmark artifacts manually.

## Full-run re-plan

At 13:53:03, 622 of 10,833 accuracy records (5.7%) had been written. GPU
utilization was 100% with 70,081 MiB free. After model startup, the observed
rate was approximately 10--12 features/second, projecting 15--17 minutes for
inference and about 20 minutes total. This remains inside the 25-minute plan,
so the schedule and 10-minute reporting reserve are unchanged.

At 13:54:45, 1,346 records (12.4%) had been written. The sustained rate under
current sharing was about 7 features/second, revising total completion to
28--30 minutes. The GPU remained at 100% utilization, so accelerator
parallelism cannot safely increase. The schedule is unchanged because the new
projection remains far inside the 110-minute execution window after reserve.

## Completion

The clean invocation ran from 13:50:01 through 14:15:25, a total of 25 minutes
24 seconds. It produced 10,833 accuracy records; LoadGen reported no warnings
or errors, and the repository evaluator completed within the reporting reserve.
