#!/usr/bin/env bash
# The M5 Max GPU throttles to ~1 TFLOPS (from ~14) when the battery is critically low, even on AC. Wait until the
# battery is back above a threshold, then run the two GPU trainings strictly one after the other (concurrent MPS
# processes thrash each other and leak wired memory).
set -u
cd "$(dirname "$0")"
MIN=${MIN_BATTERY:-40}
while true; do
  pct=$(pmset -g batt | grep -oE '[0-9]+%' | head -1 | tr -d '%')
  echo "$(date '+%H:%M') battery ${pct}%" >> results/run_when_charged.log
  [ "${pct:-0}" -ge "$MIN" ] && break
  sleep 300
done
echo "$(date '+%H:%M') starting E1 fine-tune" >> results/run_when_charged.log
SKIP_ZEROSHOT=1 STEPS=3000 P=16 TAG=indicbert_v2 ../.venv/bin/python -W ignore e1_finetune.py > results/e1_finetune_v2.log 2>&1
echo "$(date '+%H:%M') starting E2 seq2seq" >> results/run_when_charged.log
EPOCHS=10 ../.venv/bin/python -W ignore e2_seq2seq.py > results/e2_seq2seq.log 2>&1
echo "$(date '+%H:%M') done" >> results/run_when_charged.log
