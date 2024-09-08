import argparse
import psutil

from utility.executor_utils import load_configuration, execute_configs, spawn_and_execute
from utility.runner_utils import make_index_config, make_dataset

from utility.index_sizes import safe_index_size, bytes_to_gib

def index_size(config, params):
    assert len(params) == 0
    index_config, dataset = make_index_config(config), make_dataset(config)
    index_size_bytes = safe_index_size(dataset, index_config)
    return {
        "index_size_bytes": index_size_bytes,
        "index_size_gib": bytes_to_gib(index_size_bytes)
    }

def latency(config, params):
    assert len(params) == 0
    NUM_RUNS = params.get("num_runs", 3)
    assert NUM_RUNS > 0
    results = []
    for _ in range(NUM_RUNS):
        results.append(spawn_and_execute("utility/latency_runner.py", config, params))
    metrics = results[0]["metrics"]
    assert all(x["metrics"] == metrics for x in results)
    return {
        "metrics": metrics,
        "tracker": [x["tracker"] for x in results]
    }

def metrics(config, params):
    run = spawn_and_execute("utility/latency_runner.py", config, params)
    return {
        "metrics": run["metrics"]
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
        "index_size": {"callback": index_size, "parallelizable": True},
        "latency": {"callback": latency, "parallelizable": False},
        "metrics": {"callback": metrics, "parallelizable": True}
    }
    execute_configs(EXEC_INFO, configs, results_file=results_file, type_=type_,
                    params=params, max_workers=MAX_WORKERS)