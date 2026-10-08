#!/bin/bash
cd ~ && export PJRT_DEVICE=TPU PYTHONPATH=~/src MHC_DEVICE=v5litepod-1
E04="--data-dir $HOME/data --steps 6000 --warmup 200 --eval-every 1000 --log-every 25 --seed 0 --lr 3e-3 --hc-layout streams --max-seconds 420"
for v in mb8 mb8sync mb16; do
  case $v in mb8) X="";; mb8sync) X="--micro-sync 1";; mb16) X="--micro-batch 16";; esac
  rm -rf ~/repro/$v; ~/venv/bin/python -m mhc_lab.train_xla --out ~/repro/$v $E04 $X > ~/repro_$v.log 2>&1
  echo "$v: $(grep -o 'first step[^,]*' ~/repro_$v.log) | $(tail -1 ~/repro_$v.log | cut -c1-140)"
done
