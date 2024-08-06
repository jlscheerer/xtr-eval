import os
from collections import OrderedDict

import jsonlines

from xtr.data.collection import Collection
from xtr.data.queries import Queries
from xtr.data.qas import Qas

LOTTE_COLLECTION_PATH = "/lfs/1/scheerer/datasets/lotte/lotte/"

def _load_lotte_collection(collection_path):
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
    collection = _load_lotte_collection(collection_path)

    queries_path = os.path.join(dataset_path, f"questions.{type_}.tsv")
    queries = _load_queries_lotte(queries_path)

    qas_path = os.path.join(dataset_path, f"qas.{type_}.jsonl")
    qas = _load_qas_lotte(qas_path)

    return collection, queries, qas
