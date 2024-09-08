import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union

import jsonlines

from xtr.datasets.collection_paths import LOTTE_COLLECTION_PATH
from xtr.datasets.dataset import Dataset, Datasplit
from xtr.data.collection import Collection
from xtr.data.queries import Queries
from xtr.data.qas import Qas

# https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz
class LoTTE(Enum):
    WRITING = "writing"
    RECREATION = "recreation"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    LIFESTYLE = "lifestyle"
    POOLED = "pooled"

LoTTEType = Union[Literal["search"], Literal["forum"]]

@dataclass
class LoTTEDataset(Dataset):
    dataset: LoTTE
    datasplit: Datasplit
    type_: LoTTEType = "search"

    def __init__(self, dataset: LoTTE, datasplit: Datasplit, type_: LoTTEType = "search"):
        super().__init__()
        self.dataset = dataset
        self.datasplit = datasplit
        self.type_ = type_

    def _name(self):
        return f"{self.dataset}.{self.type_}.split={self.datasplit}"

    def _load(self):
        return load_lotte(self.dataset.value, self.datasplit, self.type_)

    def _eval(self, expected, rankings):
        K_VALUES = [5, 10, 100, 1000]
        final_metrics = dict()
        for k in K_VALUES:
            final_metrics[f"Success@{k}"] = _success_at_k_lotte(expected=expected, rankings=rankings, k=k)
            final_metrics[f"Recall@{k}"] = _recall_at_k_lotte(expected=expected, rankings=rankings, k=k)

def _success_at_k_lotte(expected, rankings, k):
    num_total_qids, success = 0, 0
    for qid, answer_pids in expected.data.items():
        num_total_qids += 1
        if qid not in rankings.data:
            print(f"WARNING: qid {qid} not found in {rankings}!", file=sys.stderr)
            continue
        qid_rankings = set(map(lambda x: x[0], rankings.data[qid][:k]))
        if len(qid_rankings.intersection(answer_pids)) > 0:
                success += 1
    return success / num_total_qids * 100

def _recall_at_k_lotte(expected, rankings, k):
    avg, num_relevant = 0, 0
    for qid, answer_pids in expected.data.items():
        if str(qid) not in rankings.data:
            print(f"WARNING: qid {qid} not found in {rankings}!", file=sys.stderr)
            continue
        relevant_count = len(answer_pids)
        if relevant_count == 0:
            continue
        num_relevant += 1
        qid_rankings = set(map(lambda x: x[0], rankings.data[str(qid)][:k]))
        correct_count = len(answer_pids & qid_rankings)
        avg += correct_count / relevant_count
    return avg / num_relevant

def _load_collection_lotte(collection_path):
    collection = []
    print("#> Loading collection from", collection_path, "...")
    with open(collection_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000*1000) == 0:
                print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

            pid, passage, *rest = line.strip('\n\r ').split('\t')
            assert pid == 'id' or int(pid) == line_idx, f"pid={pid}, line_idx={line_idx}"

            if len(rest) >= 1:
                title = rest[0]
                passage = title + ' | ' + passage
            collection.append(passage)

    print()
    return Collection.cast(collection)

def _load_queries_lotte(queries_path):
    queries = OrderedDict()
    print("#> Loading the queries from", queries_path, "...")
    with open(queries_path) as f:
        for line in f:
            qid, query, *_ = line.strip().split('\t')
            qid = int(qid)

            assert (qid not in queries), ("Query QID", qid, "is repeated!")
            queries[qid] = query
    print("#> Got", len(queries), "queries. All QIDs are unique.")
    return Queries.cast(dict(queries))

def _load_qas_lotte(qas_path):
    qas = OrderedDict()
    num_total_qids = 0
    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            num_total_qids += 1
            answer_pids = set(line["answer_pids"])
            qas[qid] = answer_pids
    return Qas(num_total_qids=num_total_qids, data=dict(qas))

def load_lotte(dataset, datasplit, type_):
    dataset_path = os.path.join(LOTTE_COLLECTION_PATH, dataset, datasplit)

    collection_path = os.path.join(dataset_path, "collection.tsv")
    collection = _load_collection_lotte(collection_path)

    queries_path = os.path.join(dataset_path, f"questions.{type_}.tsv")
    queries = _load_queries_lotte(queries_path)

    qas_path = os.path.join(dataset_path, f"qas.{type_}.jsonl")
    qas = _load_qas_lotte(qas_path)

    return collection, queries, qas
