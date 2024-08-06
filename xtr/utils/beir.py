import os
from enum import Enum
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

import pytrec_eval

from xtr.data.collection import MappedCollection
from xtr.data.queries import Queries
from xtr.data.rankings import Rankings
from xtr.data.qrels import Qrels

BEIR_DATASET_PATH = "/lfs/1/scheerer/datasets/beir/datasets/"

class BEIR(Enum):
    SCIFACT = 1
    NFCORPUS = 2

def load_beir(dataset: BEIR, datasplit: str):
    assert datasplit in ["train", "test", "dev"]
    
    # TODO(jlscheerer) Add a "download if required" option.

    dataset_name = {
        BEIR.SCIFACT: "scifact",
        BEIR.NFCORPUS: "nfcorpus"
    }[dataset]
    dataset_path = os.path.join(BEIR_DATASET_PATH, dataset_name)

    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split=datasplit)
    
    print(f"#> Preparing corpus for BEIR {dataset}/{datasplit}")
    documents = []
    keys = []
    for key, document in tqdm(corpus.items()):
        keys.append(key)
        documents.append(f"{document['title']} {document['text']}")

    collection = MappedCollection(documents=documents, keys=keys)
    return collection, Queries.cast(queries), Qrels.cast(qrels)

# Source: https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
def eval_metrics_beir(qrels: Qrels, rankings: Rankings):
    K_VALUES = [5, 10, 50, 100]
    METRIC_NAMES = ['ndcg_cut', 'map_cut', 'recall']

    measurements = []
    for metric_name in METRIC_NAMES:
        measurements.append(
            f"{metric_name}." + ",".join([str(k) for k in K_VALUES])
        )
    evaluator = pytrec_eval.RelevanceEvaluator(qrels.data, measurements)
    final_scores = evaluator.evaluate(rankings.flatten())

    final_metrics = dict()
    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = 0.0

    for query_id in final_scores.keys():
        for metric_name in METRIC_NAMES:
            for k in K_VALUES:
                final_metrics[f"{metric_name}@{k}"] += final_scores[query_id][
                    f"{metric_name}_{k}"
                ]

    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = round(
                final_metrics[f"{metric_name}@{k}"] / len(final_scores), 5
            )

    print("[Result]")
    for metric_name, metric_score in final_metrics.items():
        print(f"{metric_name}: {metric_score:.4f}")