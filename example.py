from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig
from xtr.modeling.xtr import XTR
from xtr.utils.beir import BEIR, load_beir, eval_metrics_beir

from xtr.utils.tracker import ExecutionTracker

collection, queries, qrels = load_beir(dataset=BEIR.SCIFACT, datasplit="test")
config = XTRConfig(index_name="beir", model=XTRModel.BASE_EN, index_config=XTRScaNNIndexConfig())
xtr = XTR(config=config, collection=collection)

tracker = ExecutionTracker(name="XTR", steps=["Query Encoding", "Candidate Generation",
                                              "Estimate Missing Similarity", "Aggregate Scores"])
rankings = xtr.retrieve_docs(queries, token_top_k=1000, document_top_k=100, tracker=tracker)
print(tracker.as_dict())

eval_metrics_beir(qrels, rankings)