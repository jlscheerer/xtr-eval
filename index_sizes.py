import os

from xtr.datasets import BEIR, BEIRDataset, LoTTE, LoTTEDataset
from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig, XTRBruteForceIndexConfig, XTRFAISSIndexConfig, XTRIndexType
from xtr.utils import xtr_tracker, canonical_index_name

def filesize(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(filesize(os.path.join(path, file)) for file in os.listdir(path))

def index_size(dataset, index_config):
    index_name = canonical_index_name(dataset=dataset, index_config=index_config)
    config = XTRConfig(index_name=index_name, model=XTRModel.BASE_EN, index_config=index_config, override=False)

    common_files = ["doc_offsets.npy", "tid2did.pickle"]
    if index_config.index_type == XTRIndexType.BRUTE_FORCE:
        specific_dir = "bruteforce"
    elif index_config.index_type == XTRIndexType.FAISS:
        specific_dir = "faiss"
    elif index_config.index_type == XTRIndexType.SCANN:
        specific_dir = "scann"
    else: raise AssertionError
    
    total_size = 0
    for entry in common_files + [specific_dir]:
        total_size += filesize(os.path.join(config.path, entry))
    return total_size

def bytes_to_gib(size):
    return size / (1024 * 1024 * 1024)

DATASETS = [BEIRDataset(dataset=BEIR.NFCORPUS, datasplit="test"),
            BEIRDataset(dataset=BEIR.SCIFACT, datasplit="test"),
            BEIRDataset(dataset=BEIR.SCIDOCS, datasplit="test"),
            BEIRDataset(dataset=BEIR.FIQA_2018, datasplit="test"),
            BEIRDataset(dataset=BEIR.TOUCHE_2020, datasplit="test"),
            BEIRDataset(dataset=BEIR.QUORA, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.LIFESTYLE, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.RECREATION, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.WRITING, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.TECHNOLOGY, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.SCIENCE, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.POOLED, datasplit="test"),]
INDEX_CONFIGS = [XTRBruteForceIndexConfig(), XTRFAISSIndexConfig(), XTRScaNNIndexConfig()]

for dataset in DATASETS:
    for index_config in INDEX_CONFIGS:
        try:
            size = bytes_to_gib(index_size(dataset, index_config))
        except:
            size = "-"
        print(dataset.name, index_config.name, size, "GiB")