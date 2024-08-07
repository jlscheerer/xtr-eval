import os
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

import pytrec_eval

from xtr.datasets.dataset import Dataset, Datasplit
from xtr.data.collection import MappedCollection
from xtr.data.queries import Queries
from xtr.data.rankings import Rankings
from xtr.data.qrels import Qrels

BEIR_COLLECTION_PATH = "/lfs/1/scheerer/datasets/beir/datasets/"

# https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets
class BEIR(Enum):
    NFCORPUS = "nfcorpus"
    FIQA_2018 = "fiqa"
    SCIDOCS = "scidocs"
    SCIFACT = "scifact"

@dataclass
class BEIRDataset(Dataset):
    dataset: BEIR
    datasplit: Datasplit

    def __init__(self, dataset: BEIR, datasplit: Datasplit):
        super().__init__()
        self.dataset = dataset
        self.datasplit = datasplit

    def _name(self):
        return f"{self.dataset}.split={self.datasplit}"

    def _load(self):
        return load_beir(self.dataset, self.datasplit)

    def _eval(self, expected, rankings):
        return eval_metrics_beir(expected, rankings)

def load_beir(dataset: BEIR, datasplit: str, create_if_not_exists: bool=True):
    assert datasplit in ["train", "test", "dev"]
    
    dataset_name = dataset.value
    dataset_path = os.path.join(BEIR_COLLECTION_PATH, dataset_name)

    if not os.path.exists(dataset_path):
        if not create_if_not_exists:
            raise AssertionError(f"Could not load BEIR dataset {dataset}!")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, BEIR_COLLECTION_PATH)
        return

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