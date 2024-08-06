class Rankings:
    def __init__(self, data=[]):
        super().__init__()
        self.data = data

    def __str__(self):
        text = self._has_document_text()
        if len(self.data) == 1:
            return self._build_results_view(self.data[0], text=text, prefix="")
        view = str()
        for q_idx, result in enumerate(self.data):
            view += f"query_id={q_idx}: \n" + self._build_results_view(result, text=text)
        return view

    def _has_document_text(self):
        if len(self.data) == 0:
            return True
        return len(self.data[0][0]) == 3

    def _build_results_view(self, result, text=False, prefix=" "):
        if text:
            return "\n".join([f"{prefix}[{rank}] doc={did} ({score:.3f}): {doc}" for rank, (did, score, doc) in enumerate(result)])
        return "\n".join([f"{prefix}[{rank}] doc={did} ({score:.3f})" for rank, (did, score) in enumerate(result)])