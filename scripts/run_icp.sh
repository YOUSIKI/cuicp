#!/bin/bash

mkdir -p data/out

for i in {1..5}; do
  src="data/cal/C$i.asc"
  tar="data/cal/C6.asc"
  out="data/out/C$i.asc"
  python icp.py \
    --source $src \
    --target $tar \
    --output $out \
    --percentile 0.5 \
    --max-steps 1000 \
    --early-stop-eps 0.0001 \
    --early-stop-steps 25
done

cp data/cal/C6.asc data/out/C6.asc
