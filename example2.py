from xtr.modeling.xtr import XTR

xtr = XTR.load_index("test_index")

query = "Who founded google"
retrieved_docs, metadata = xtr.retrieve_docs(query, document_top_k=3, return_text=True)

print(f"\nQuery: {query}")
for rank, (did, score, doc) in enumerate(retrieved_docs[0]):
    print(f"[{rank}] doc={did} ({score:.3f}): {doc}")