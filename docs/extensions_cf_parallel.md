# Optional CF process scheduling

`scripts/run_extension_cf_parallel.py` schedules one frozen Stage B CF family
using two spawned processes, each with one numerical thread. It calls the
existing training, resume, evaluation, extension-decision and selection
functions without modifying their code or family protocol.

```bash
# Stop an existing serial CF runner first; do not launch it during this run.
.venv/bin/python scripts/run_extension_cf_parallel.py --method lightgcn_user

# Separate phases are also supported.
.venv/bin/python scripts/run_extension_cf_parallel.py --method lightgcn_user --hpo-only
.venv/bin/python scripts/run_extension_cf_parallel.py --method lightgcn_user --final-only
```

The runner refuses an active `run_extension_cf.py` in this checkout and uses
an exclusive lock against another parallel runner. The legacy CLI does not
implement that lock, so it must remain stopped throughout the handoff and
execution. Existing canonical completion receipts are verified and reused;
an incomplete trajectory resumes through the existing Adam/RNG state checks.

All 36 historical trajectories run to 50 epochs. The original three-seed
selection rule decides whether all 36 extend to 100. The original `run_hpo`
then rechecks the matrix and writes the canonical selected configuration.
Each of the 15 final tasks validates that selection before loading its final
prefix. No scheduling decision uses final-query metrics. The original final
query gate and per-trajectory artifact schemas remain unchanged.

Every invocation freezes a separate scheduling receipt under
`<family>/parallel_execution/`, including the scheduler hash, worker count,
family/global protocol identities, input hashes and timestamps. This receipt
records an execution change, not a new research protocol or a replacement
for original model completion receipts. Failure receipts remain available.

Wall times can differ with contention. A worker's peak RSS is its process
high-water mark across jobs; it must not be summed or interpreted as an
independent model memory requirement. The serial entry point remains a valid
way to reproduce the same frozen experiment.

For long jobs in a remote session, redirect output to a persistent file so a
closed interactive output pipe cannot interrupt the training callback:

```bash
mkdir -p logs_extensions
nohup .venv/bin/python -u scripts/run_extension_cf_parallel.py \
  --method lightgcn_user --retry-failed \
  > logs_extensions/cf_parallel.log 2>&1 &
```

A retry verifies the existing epoch states and retains resolved failure records.
The completed trajectory and final-model receipts, rather than process exit or
log messages alone, determine whether the experiment is complete.
