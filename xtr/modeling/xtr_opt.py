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

    def _get_token_embeddings(self, texts, maxlen):
        batch_embeds = self.encoder([t.lower() for t in texts], maxlen=maxlen)
        batch_lengths = torch.sum(batch_embeds["mask"].cpu(), axis=1)
        return batch_embeds["encodings"].cpu(), batch_lengths

    def _get_flatten_embeddings(self, batch_text, maxlen, return_last_offset=False):
        batch_embeddings, batch_lengths = self._get_token_embeddings(batch_text, maxlen=maxlen)

        nqueries = batch_lengths.flatten().shape[0]
        offsets = torch.zeros(nqueries + return_last_offset, dtype=torch.int32)
        torch.cumsum(batch_lengths.flatten()[:(None if return_last_offset else -1)], dim=0, out=offsets[1:])

        indices = torch.arange(maxlen).expand(nqueries, maxlen)
        flatten_embeddings = batch_embeddings[indices < batch_lengths].view(-1, 128)
        
        return flatten_embeddings, offsets

    def _batch_search_tokens(self, batch_query, token_top_k, leaves_to_search, pre_reorder_num_neighbors, tracker):
        assert len(batch_query) == 1
        
        tracker.begin("Query Encoding")
        query_encodings, query_offsets = self._get_flatten_embeddings(batch_query, maxlen=self.config.query_maxlen, return_last_offset=True)
        tracker.end("Query Encoding")

        tracker.begin("search_batched")
        neighbors, scores = self.searcher.search_batched(
            query_encodings, final_num_neighbors=token_top_k, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors
        )
        tracker.end("search_batched")

        tracker.begin("enumerate_scores")
        tracker.end("enumerate_scores")
        
        return neighbors, torch.from_numpy(scores)

    def _estimate_missing_similarity(self, batch_result, tracker):
        neighbors, scores = batch_result
        ntokens, _ = neighbors.shape

        tracker.begin("Estimate Missing Similarity")
        
        ems_vector = torch.zeros(self.config.query_maxlen, dtype=torch.float32)

        # Use similarity of the last token as imputed similarity.
        ems_vector[:ntokens] = scores[:, -1]
        
        tracker.end("Estimate Missing Similarity")
        
        return ems_vector

    def _aggregate_scores(self, batch_result, ems_vector, document_top_k, tracker):
        """Aggregates token-level retrieval scores into query-document scores."""
        assert ems_vector.shape == (self.config.query_maxlen,)
        neighbors, scores = batch_result

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
        flat_score_matrix, _ = scatter_max(scores.flatten(), indices, out=score_matrix.view(-1))
        score_matrix = flat_score_matrix.view(score_matrix.size())
        
        tracker.end("get_did2scores")

        tracker.begin("add_ems")

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