#!/usr/bin/bash

# Build ScaNN Indexes for BEIR/test
BEIR=("nfcorpus" "fiqa" "scidocs" "scifact")
for dataset in "${BEIR[@]}"; do
    python utils.py index -c beir -d "$dataset" -s test -i scann
done

# Build ScaNN Indexes for LoTTE.search/test
LoTTE=("writing" "recreation" "science" "technology" "lifestyle" "pooled")
for dataset in "${LoTTE[@]}"; do
    python utils.py index -c lotte -d "$dataset" -t search -s test -i scann
done