from xtr.utils.beir import BEIR, load_beir

collection, queries, qrels = load_beir(dataset=BEIR.SCIFACT, datasplit="test")

print(collection)