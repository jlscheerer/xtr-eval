from xtr.datasets import BEIR, BEIRDataset, LoTTE, LoTTEDataset
from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig, XTRBruteForceIndexConfig, XTRFAISSIndexConfig, XTRIndexType

def make_index_config(config):
    if config["index_type"] == "bruteforce":
        return XTRBruteForceIndexConfig()
    elif config["index_type"] == "faiss":
        return XTRFAISSIndexConfig()
    elif config["index_type"] == "scann":
        return XTRScaNNIndexConfig()
    assert False

def make_dataset(config):
    collection, dataset, split = config["collection"], config["dataset"], config["split"]
    if collection == "beir":
        return BEIRDataset(dataset=BEIR(dataset), datasplit=split)
    elif collection == "lotte":
        return LoTTEDataset(dataset=LoTTE(dataset), datasplit=split)
    assert False