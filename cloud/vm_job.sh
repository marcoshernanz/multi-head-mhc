#!/bin/bash
# On a TPU VM: run one training job on one chip, detached from the ssh session that starts it.
# Usage: vm_job.sh CHIP NCHIPS ACCEL JOB_NAME -- train_xla args...
# Writes ~/runs/JOB/{run.json,metrics.jsonl,summary.json,log.txt,ckpt.pt} and, when the process ends for any reason, exit_code.
# A job restarted on the same VM resumes from its ckpt.pt (--ckpt-every); a fresh VM after a preemption starts it from step 0.
CHIP=$1; NCHIPS=$2; ACCEL=$3; JOB=$4; shift 5
RUN="$HOME/runs/$JOB"
mkdir -p "$RUN"
rm -f "$RUN/exit_code"
export PJRT_DEVICE=TPU PYTHONPATH="$HOME/src" MHC_DEVICE="$ACCEL"
if [ "$NCHIPS" -gt 1 ]; then  # one process per chip: the "visible_chips" recipe that worked on Kaggle's v5e-8
  PORT=$((8476 + CHIP))
  export TPU_VISIBLE_CHIPS=$CHIP TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
         TPU_PROCESS_PORT=$PORT TPU_PROCESS_ADDRESSES=localhost:$PORT TPU_RUNTIME_METRICS_PORTS=$((8431 + CHIP)) CLOUD_TPU_TASK_ID=0
fi
echo "$$" > "$RUN/pid"
echo "$CHIP" > "$RUN/chip"
"$HOME/venv/bin/python" -m mhc_lab.train_xla --out "$RUN" --data-dir "$HOME/data" "$@" >> "$RUN/log.txt" 2>&1
echo $? > "$RUN/exit_code"
