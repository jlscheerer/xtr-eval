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

# TODO(jlscheerer) Merge with the previous loop.
# Build FAISS Indexes for BEIR/test
BEIR=("nfcorpus" "fiqa" "scidocs" "scifact")
for dataset in "${BEIR[@]}"; do
    python utils.py index -c beir -d "$dataset" -s test -i faiss
done

# Build FAISS Indexes for LoTTE.search/test (subset)
python utils.py index -c lotte -d writing -t search -s test -i faiss
python utils.py index -c lotte -d lifestyle -t search -s test -i faiss

# Build BruteForce Indexes for BEIR/test
BEIR=("nfcorpus" "fiqa" "scidocs" "scifact")
for dataset in "${BEIR[@]}"; do
    python utils.py index -c beir -d "$dataset" -s test -i bruteforce
done

# Build BruteForces Indexes for LoTTE.search/test (subset)
python utils.py index -c lotte -d writing -t search -s test -i bruteforce
python utils.py index -c lotte -d lifestyle -t search -s test -i bruteforce