import os
from enum import Enum
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from xtr.data.collection import MappedCollection
from xtr.data.queries import Queries
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