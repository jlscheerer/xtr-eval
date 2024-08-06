from xtr.utils.lotte import load_lotte

collection, queries, qas = load_lotte(dataset="lifestyle", datasplit="test", type_="search")

print(collection, queries, qas)