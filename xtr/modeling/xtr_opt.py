import torch
import numpy as np
from torch_scatter import scatter_max

from tqdm import tqdm

from xtr.modeling.xtr import XTR
from xtr.config import XTRConfig
from xtr.data.collection import Collection


class XTROpt(XTR):
    def __init__(self, config: XTRConfig, collection: Collection, device=torch.device("cuda")):
        super().__init__(config=config, collection=collection, device=device)

        # TODO(jlscheerer) Perhaps we should cache this?
        ntokens = len(self.tid2did) - 1 # tid2did contains -1 for invalid
        self.tid2did_vectorized = torch.zeros(ntokens, dtype=torch.int32)

        for key, value in tqdm(self.tid2did.items()):
            if key == -1:
                continue
            self.tid2did_vectorized[key] = value

    def _aggregate_scores(self, batch_result, batch_ems, document_top_k, tracker):
        """Aggregates token-level retrieval scores into query-document scores."""
        assert len(batch_result) == 1 and len(batch_ems) == 1

        # TODO(jlscheerer) this is an extremely weird way of storing this!
        # neighbors: (ntokens, token_top_k), represents the token_ids of the retrieved "candidates"
        query_tokens, neighbors, scores = batch_result[0]

        assert neighbors.shape == scores.shape
        ntokens, token_top_k = neighbors.shape
        
        tracker.begin("get_did2scores")

        # Total number of unique documents.
        candidate_dids = self.tid2did_vectorized[neighbors]
        idx_to_candidate_did = torch.unique(candidate_dids.flatten(), sorted=True)
        ncandidates = idx_to_candidate_did.shape[0]

        # Construct a tensor indicating the qtoken_idx for each candidate.
        candidate_qids = torch.repeat_interleave(torch.arange(ntokens), token_top_k)

        # Construct a tensor indicating the index corresponding to each of the candidate document ids.
        candidate_dids_idx = torch.searchsorted(idx_to_candidate_did, candidate_dids.flatten())

        # *Exact* index into the flattened score matrix for (candidate_did, token_idx)
        indices = candidate_dids_idx * self.config.query_maxlen + candidate_qids

        # The score_matrix corresponds to did2scores dictionary.
        score_matrix = torch.zeros((ncandidates, self.config.query_maxlen), dtype=torch.float)

        # Populate the score matrix using the maximum value for each index.
        flat_score_matrix, _ = scatter_max(torch.from_numpy(scores.flatten()), indices, out=score_matrix.view(-1))
        score_matrix = flat_score_matrix.view(score_matrix.size())
        
        tracker.end("get_did2scores")

        tracker.begin("add_ems")

        # TODO(jlscheerer) Adapt this once we compute batch_ems more efficiently.
        # i.e., we should directly emit this instead of changing the format here.
        # print(batch_ems)
        ems_vector = torch.zeros(self.config.query_maxlen, dtype=torch.float32)
        for (idx, _), score in batch_ems[0].items():
            ems_vector[idx] = score.item()

        # Apply the ems values column-wise (by replacing zeros).
        ems_matrix = ems_vector.view(1, -1).expand_as(score_matrix)
        zero_mask = score_matrix == 0
        score_matrix[zero_mask] = ems_matrix[zero_mask]

        tracker.end("add_ems")

        tracker.begin("get_final_score")

        # Compute the per passage score by summing across rows.
        # NOTE to be consistent with the original implementation we "average" across the number of tokens
        scores = score_matrix.sum(dim=1) / ntokens

        tracker.end("get_final_score")

        tracker.begin("sort_scores")

        # TODO(jlscheerer) We could switch to top-k for small document_top_k values.
        scores, indices = torch.sort(scores, stable=True, descending=True)
        document_ids = idx_to_candidate_did[indices]
        
        # Construct the batch_ranking from the vectorized implementation
        batch_ranking = [[(docid.item(), score.item()) for docid, score in zip(document_ids[:document_top_k], scores[:document_top_k])]]
        
        tracker.end("sort_scores")
        
        return batch_ranking