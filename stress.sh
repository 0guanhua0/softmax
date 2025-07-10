#!/bin/bash
set -euo pipefail

i=1
while true; do
  echo "--- run $i ---"
  python benchmark.py 2>&1

  if [[ $? -ne 0 ]]; then
    echo "fail on run $i"
    break
  fi

  ((i++))
done | tee -a stress.log
