# Experiment history and artifact provenance

The earlier review found two failures that motivated the documented revision:

1. Environment version 2: seed 42 met task-outcome gates but failed the probe
   requiring frustration to be acknowledged before proceeding.
2. Environment version 3, initial 50k segment: both seeds passed the emotion
   probes, but seed 7 timed out on a caller with no matching claim.

Version 3 introduced a small cost for ignoring observable distress. Both seeds
then received the same additional 25k segment with unchanged acceptance gates.
Reward totals across environment versions are not directly comparable.

During delivery, iCloud replaced parts of the Desktop workspace with unreadable
placeholder files. The original workspace was retained. A local recovery used
remote commit `989b088` and the successful patches recorded during this task.
The reconstructed training source and fixture hash exactly matched the hash
captured before the controlled emotion experiment:

`09c6d4738f568b47f2cd68809cdd94e912212da0849db2bd8b140a1213488f25`

The version-2 binary archive was not recoverable into this copy; the first item
above is a record of the earlier observed failure, not a claim that its original
checkpoint is included. Current delivery checkpoints and the version-3 50k
intermediate evidence are regenerated using the documented seeds and budgets.
Their own hashes and audit results identify these new files. The four emotion
ablation runs and browser evidence were recovered directly as readable files.

Use `artifacts/rl_audit.json` for current acceptance. Historical observations do
not substitute for rerunning the delivered checkpoints, and a failed run is not
removed merely because it misses an acceptance gate.
