#!/bin/bash
# Poll a Kaggle kernel until it finishes; print the final status line.
slug=$1
while true; do
  s=$(kaggle kernels status "marcoshernanz/$slug" 2>&1 | grep -v outdated)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*|*FAIL*) echo "$s"; exit 0;;
  esac
  sleep 60
done
