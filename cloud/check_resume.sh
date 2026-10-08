#!/bin/bash
# On a TPU VM: run A uninterrupted; run B killed after step ~230 and resumed (checkpoints every 100 steps); every step logged.
# Then: B vs A must be bit-identical if resuming is exact and the TPU is deterministic.
set -u
cd "$HOME"
export PJRT_DEVICE=TPU PYTHONPATH="$HOME/src" MHC_DEVICE=v6e
ARGS="--data-dir $HOME/data --steps 300 --warmup 200 --log-every 1 --eval-every 100 --eval-batches 5 --final-eval-batches 10
      --hc-layout streams --hc-heads 4 --seed 0 --lr 3e-3 --ckpt-every 100"
rm -rf "$HOME/check/A" "$HOME/check/B"
"$HOME/venv/bin/python" -m mhc_lab.train_xla --out "$HOME/check/A" $ARGS > "$HOME/check_A.log" 2>&1
"$HOME/venv/bin/python" -m mhc_lab.train_xla --out "$HOME/check/B" $ARGS > "$HOME/check_B1.log" 2>&1 &
PID=$!
until grep -q '"step": 230' "$HOME/check/B/metrics.jsonl" 2>/dev/null; do sleep 1; done
kill -9 $PID; wait $PID 2>/dev/null
echo "killed B at: $(tail -1 "$HOME/check/B/metrics.jsonl" | cut -c1-40)"
"$HOME/venv/bin/python" -m mhc_lab.train_xla --out "$HOME/check/B" $ARGS > "$HOME/check_B2.log" 2>&1
grep -h resumed "$HOME/check_B2.log"
"$HOME/venv/bin/python" "$HOME/check_resume.py" "$HOME/check/A" "$HOME/check/B"
