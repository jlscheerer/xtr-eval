import os
import json
import re

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

LABEL = "scann"
OPT = False

def _extract_run_config(run):
    return {
        "config": run["config"],
        "metrics": run["metrics"]
    }

def _extract_tracker_stats(run):
    tracker = run["tracker"]
    return {
        "name": tracker["name"],
        "steps": tracker["steps"],
        "num_iterations": tracker["num_iterations"]
    }

def _extract_tracker_measurements(run):
    tracker = run["tracker"]
    return {
        "time_per_step": tracker["time_per_step"],
        "iteration_time": tracker["iteration_time"]
    }

def _aggregate_latency_results(runs):
    assert len(runs) == 3

    results = []
    for data in zip(*runs):
        config = _extract_run_config(data[0])
        tracker = _extract_tracker_stats(data[0])
        
        # Sanity check: run configurations should be identical.
        assert all(config == _extract_run_config(run) for run in data)
        assert all(tracker == _extract_tracker_stats(run) for run in data)

        times = [_extract_tracker_measurements(run)["time_per_step"] for run in data]

        # Minimum average latency across three runs
        min_times = dict()
        for step in tracker["steps"]:
            min_times[step] = min(time[step] for time in times)

        results.append({
            "config": config,
            "num_iterations": tracker["num_iterations"],
            "time_per_step": min_times
        })
    return results

def _model_config_name(config):
    config = config["config"]
    assert config["index"] == "XTRIndexType.SCANN"

    return f"XTR/ScaNN (document_top_k={config['document_top_k']}, token_top_k={config['token_top_k']})"

def _dataset_config_name(config):
    match = re.match("LoTTE\\.(.*)\\.search\\.split\\=(.*)", config["config"]["dataset"])
    if match is not None:
        assert match[2] == "test"
        dataset = match[1].title()
        return f"LoTTE {dataset}"
    match = re.match("BEIR\\.(.*)\\.split\\=(.*)", config["config"]["dataset"])
    assert match is not None
    assert match[2] == "test"
    dataset = {
        "SCIFACT": "SciFact"
    }[match[1]]
    return f"BEIR {dataset}"

def _config_name(config):
    return f"{_model_config_name(config)}, {_dataset_config_name(config)}"

def _latency_breakdown(latency):
    num_iterations = latency["num_iterations"]
    return [(key, value / num_iterations) for key, value in latency["time_per_step"].items()]

def _render_aggregated_latency_results(latencies, name):
    totals = []
    for latency in latencies:
        breakdown = _latency_breakdown(latency)
        totals.append(sum([x[1] for x in breakdown]) * 1000)
    bound = round(max(*totals) / 100) * 100

    pdf = PdfPages(os.path.join("results", name))
    for latency in latencies:
        name = _config_name(latency["config"])
        num_iterations = latency["num_iterations"]
        breakdown = _latency_breakdown(latency)
        df = pd.DataFrame(
            {
                "Task": [x[0] for x in breakdown],
                "Duration": [x[1] * 1000 for x in breakdown],
            }
        )
        df["Start"] = df["Duration"].cumsum().shift(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 2.5))

        for i, task in enumerate(df["Task"]):
            start = df["Start"][i]
            duration = df["Duration"][i]
            ax.barh("Tasks", duration, left=start, height=0.5, label=task)

        plt.xlabel("Latency (ms)")
        accumulated = round(sum([x[1] for x in breakdown]) * 1000, 1)
        plt.title(
            f"{name} (iterations={num_iterations}, total={accumulated}ms)"
        )
        ax.set_yticks([])
        ax.set_ylabel("")
        if bound is not None:
            ax.set_xlim([0, bound])

        plt.tight_layout()
        plt.legend()
        # plt.savefig("plot.pdf")
        pdf.savefig()
    
    pdf.close()

if __name__ == "__main__":
    base_name = f"run_xtr.{LABEL}{'.opt' if OPT else ''}_"
    pattern = f"{base_name}\\d{{4}}(-\\d{{2}}){{5}}\\.json"
    files = [filename for filename in os.listdir("results") if re.match(pattern, filename)]
    assert len(files) == 3
    runs = []
    for filename in files:
        with open(os.path.join("results", filename), "r") as file:
            runs.append(json.load(file))
    latencies = _aggregate_latency_results(runs)
    name = f"results_{LABEL}{'.opt' if OPT else ''}.pdf"
    _render_aggregated_latency_results(latencies, name)