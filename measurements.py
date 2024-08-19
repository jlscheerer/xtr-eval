from xtr.experiments import *

LABEL = "xtr.scann"
DATASETS = [BEIRDataset(dataset=BEIR.SCIFACT, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.LIFESTYLE, datasplit="test"),
            LoTTEDataset(dataset=LoTTE.TECHNOLOGY, datasplit="test")]
INDEX_CONFIGS = [XTRScaNNIndexConfig()]
TOKEN_TOP_K_VALUES = [1_000, 40_000]

for _ in range(NUM_RUNS_PER_EXPERIMENT):
    xtr_run_configurations(datasets=DATASETS, index_configs=INDEX_CONFIGS,
                           document_top_k=100, token_top_k_values=TOKEN_TOP_K_VALUES, label=LABEL)