class Rankings:
    def __init__(self, data, index_map: None, text: bool=False):
        super().__init__()
        self.data = {
            key: Rankings._remap_document_indices(index_map, value)
            for key, value in data.items()
        }
        self.text = text

    def flatten(self):
        return {
            qidx: {
                entry[0]: entry[1] for entry in result
            }
            for qidx, result in self.data.items()
        }

    def __str__(self):
        if len(self.data) == 1:
            return self._build_results_view(self.data[0], text=self.text, prefix="")
        view = "\n"
        for q_idx, result in self.data.items():
            view += f"query_id={q_idx}: \n" + self._build_results_view(result, text=self.text) + "\n"
        return view

    def _build_results_view(self, result, text=False, prefix=" "):
        if text:
            return "\n".join([f"{prefix}[{rank}] doc={did} ({score:.3f}): {doc}" for rank, (did, score, doc) in enumerate(result)])
        return "\n".join([f"{prefix}[{rank}] doc={did} ({score:.3f})" for rank, (did, score) in enumerate(result)])

    @staticmethod
    def _remap_document_indices(index_map, result):
        if index_map is None:
            return result
        remapped = []
        for row in result:
            # Remap the did's based on the provided `index_map`.
            remapped.append((index_map[row[0]], *row[1:]))
        return remapped