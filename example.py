from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig
from xtr.modeling.xtr import XTR
from xtr.utils.lotte import load_lotte
from xtr.utils.tracker import ExecutionTracker

collection, queries, qas = load_lotte(dataset="lifestyle", datasplit="test", type_="search")
config = XTRConfig(index_name="lotte", model=XTRModel.BASE_EN, index_config=XTRScaNNIndexConfig())
xtr = XTR(config=config, collection=collection)

tracker = ExecutionTracker(name="XTR", steps=["Query Encoding", "Candidate Generation",
                                              "Estimate Missing Similarity", "Aggregate Scores"])
rankings = xtr.retrieve_docs(queries, token_top_k=1000, document_top_k=100, tracker=tracker)
print(tracker.as_dict())