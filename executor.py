import argparse
import psutil

from xtr.datasets import BEIR, BEIRDataset, LoTTE, LoTTEDataset
from utility.executor_utils import load_configuration, execute_configs
from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig, XTRBruteForceIndexConfig, XTRFAISSIndexConfig, XTRIndexType

from utility.index_sizes import safe_index_size, bytes_to_gib

def _make_index_config(config):
    if config["index_type"] == "bruteforce":
        return XTRBruteForceIndexConfig()
    elif config["index_type"] == "faiss":
        return XTRFAISSIndexConfig()
    elif config["index_type"] == "scann":
        return XTRScaNNIndexConfig()
    assert False

def _make_dataset(config):
    collection, dataset, split = config["collection"], config["dataset"], config["split"]
    if collection == "beir":
        return BEIRDataset(dataset=BEIR(dataset), datasplit=split)
    elif collection == "lotte":
        return LoTTEDataset(dataset=LoTTE(dataset), datasplit=split)
    assert False

def index_size(config):
    index_config, dataset = _make_index_config(config), _make_dataset(config)
    index_size_bytes = safe_index_size(dataset, index_config)
    return {
        "index_size_bytes": index_size_bytes,
        "index_size_gib": bytes_to_gib(index_size_bytes)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='xtr-eval Experiment [Executor/Platform]')
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers or psutil.cpu_count(logical=False)
    OVERWRITE = args.overwrite
    results_file, type_, params, configs = load_configuration(args.config, overwrite=OVERWRITE)

    EXEC_INFO = {
        "index_size": {"callback": index_size, "parallelizable": True}
    }
    execute_configs(EXEC_INFO, configs, results_file=results_file, type_=type_,
                    params=params, max_workers=MAX_WORKERS)