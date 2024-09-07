import os
import json

import os
import io
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout

BEIR_DATASETS = ["nfcorpus", "scifact", "scidocs", "fiqa", "webis-touche2020", "quora"]
LOTTE_DATASETS = ["lifestyle", "writing", "recreation", "technology", "science", "pooled"]

def _make_config(collection, dataset, index_type, opt, document_top_k, token_top_k, split="test"):
    assert collection in ["beir", "lotte"]
    if collection == "beir":
        assert dataset in BEIR_DATASETS
    else:
        assert dataset in LOTTE_DATASETS
    assert split in ["dev", "test"]
    assert index_type in ["bruteforce", "faiss", "scann"]
    assert document_top_k is None or (isinstance(document_top_k, int) and document_top_k > 0)
    assert token_top_k is None or (isinstance(token_top_k, int) and token_top_k > 0)
    assert opt is None or isinstance(opt, bool)
    return {
        "collection": collection,
        "dataset": dataset,
        "index_type": index_type,
        "optimized": opt,
        "document_top_k": document_top_k,
        "token_top_k": token_top_k,
        "split": split
    }

def _expand_configs(datasets, index_types, optimized, document_top_ks, token_top_ks, split="test"):
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(index_types, list):
        index_types = [index_types]
    if not isinstance(optimized, list):
        optimized = [optimized]
    if not isinstance(document_top_ks, list):
        document_top_ks = [document_top_ks]
    if not isinstance(token_top_ks, list):
        token_top_ks = [token_top_ks]
    configs = []
    for collection_dataset in datasets:
        collection, dataset = collection_dataset.split(".")
        for index_type in index_types:
            for opt in optimized:
                for document_top_k in document_top_ks:
                    for token_top_k in token_top_ks:
                        configs.append(_make_config(collection=collection, dataset=dataset, index_type=index_type, opt=opt,
                                                document_top_k=document_top_k, token_top_k=token_top_k, split=split))
    return configs

def _get(config, key):
    if key in config:
        return config[key]
    return None

def _expand_configs_file(configuration_file):
    configs = configuration_file["configurations"]
    return _expand_configs(datasets=_get(configs, "datasets"), index_types=_get(configs, "index_type"),
                           optimized=_get(configs, "optimized"),document_top_ks=_get(configs, "document_top_ks"),
                           token_top_ks=_get(configs, "token_top_ks"), split=_get(configs, "datasplit"))

def _write_results(results_file, data):
    with open(results_file, "w") as file:
        file.write(json.dumps(data, indent=3))

def load_configuration(filename, overwrite=False):
    with open(filename, "r") as file:
        config_file = json.loads(file.read())
    name = config_file["name"]
    type_ = config_file["type"]
    params = _get(config_file, "parameters") or {}
    configs = _expand_configs_file(config_file)

    os.makedirs("experiments/results", exist_ok=True)
    results_file = os.path.join("experiments/results", f"{name}.json")
    assert not os.path.exists(results_file) or overwrite

    _write_results(results_file, [])
    return results_file, type_, params, configs

def _init_proc(env_vars):
    for key, value in env_vars.items():
        os.environ[key] = value

def _execute_configs_parallel(configs, callback, type_, results_file, max_workers):
    env_vars = dict(os.environ)
    progress = tqdm(total=len(configs))
    results = []
    with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_proc, initargs=(env_vars,)
        ) as executor, redirect_stdout(
            io.StringIO()
        ) as rd_stdout:
        futures = {executor.submit(callback, config): config for config in configs}
        for future in as_completed(futures.keys()):
            result = future.result()
            config = futures[future]

            result["provenance"] = config
            result["provenance"]["type"] = type_
            results.append(result)
            _write_results(results_file=results_file, data=results)
            
            sys.stdout = sys.__stdout__
            sys.stdout = rd_stdout
            progress.update(1)
    progress.close()

def _execute_configs_sequential(configs, callback, type_, results_file):
    results = []
    for config in tqdm(configs):
        result = callback(config)
        result["provenance"] = config
        result["provenance"]["type"] = type_
        results.append(result)
        _write_results(results_file=results_file, data=results)

def execute_configs(exec_info, configs, results_file, type_, params, max_workers):
    exec_info = exec_info[type_]
    callback, parallelizable = exec_info["callback"], exec_info["parallelizable"]
    if parallelizable:
        _execute_configs_parallel(configs, callback, type_, results_file, max_workers=max_workers)
    else:
        _execute_configs_sequential(configs, callback, type_, results_file)