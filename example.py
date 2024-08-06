from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig
from xtr.modeling.xtr import XTR
from xtr.utils.beir import BEIR, load_beir

collection, queries, qrels = load_beir(dataset=BEIR.SCIFACT, datasplit="test")

config = XTRConfig(index_name="beir", model=XTRModel.BASE_EN, index_config=XTRScaNNIndexConfig(), override=True)
xtr = XTR(config=config, collection=collection)

rankings = xtr.retrieve_docs(["0-dimensional biomaterials show inductive properties."], token_top_k=1000, document_top_k=10)

print(rankings)