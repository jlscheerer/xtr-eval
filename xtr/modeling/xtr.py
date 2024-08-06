import os
import json
import shutil
import pickle
from tqdm import tqdm
from typing import Optional, Union, List

import numpy as np
import tensorflow as tf

import scann
import faiss

from xtr.config import XTRConfig, XTRIndexType, XTRScaNNIndexConfig, XTRFAISSIndexConfig, XTRBruteForceIndexConfig
from xtr.modeling.encoder import XTREncoder

# Adapted from: https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
class XTR(object):
    def __init__(self, config: XTRConfig):
        self.config = config

        if not config.is_tpu():
            assert config.index_type == XTRIndexType.SCANN or config.index_type == XTRIndexType.BRUTE_FORCE
            physical_devices = tf.config.list_physical_devices("GPU")
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"set_memory_growth = True for {gpu}")
                if len(physical_devices) == 0:
                    print("Loading XTR on CPU.")
            except Exception as e:
                print(e)
        else:
            assert config.index_type == XTRIndexType.FAISS or config.index_type == XTRIndexType.BRUTE_FORCE
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                tf.config.experimental.enable_mlir_bridge()
            except Exception as e:
                print(e)

        self.encoder = XTREncoder(config)

    def _get_token_embeddings(self, texts):
        batch_embeds = self.encoder([t.lower() for t in texts])
        batch_lengths = np.sum(batch_embeds["mask"].numpy(), axis=1)
        return batch_embeds["encodings"].cpu().numpy(), batch_lengths

    def _get_flatten_embeddings(self, batch_text, return_last_offset=False):
        batch_embeddings, batch_lengths = self._get_token_embeddings(batch_text)
        flatten_embeddings = None
        num_tokens = 0
        offsets = [0]
        for _, (embeddings, length) in enumerate(zip(batch_embeddings, batch_lengths)):
            if flatten_embeddings is not None:
                flatten_embeddings = np.append(flatten_embeddings, embeddings[:int(length)], axis=0)
            else:
                flatten_embeddings = embeddings[:int(length)]
            num_tokens += int(length)
            offsets.append(num_tokens)
        assert num_tokens == flatten_embeddings.shape[0]
        if not return_last_offset:
            offsets = offsets[:-1]
        return flatten_embeddings, offsets

    def build_index(self, documents, batch_size=32):
        all_token_embeds = np.zeros((len(documents)*self.config.max_seq_len, self.config.token_embed_dim), dtype=np.float32)
        all_doc_offsets = []
        num_tokens = 0
        for batch_idx in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[batch_idx:batch_idx+batch_size]
            batch_embeds, batch_offsets = self._get_flatten_embeddings(batch_docs)
            all_doc_offsets += [num_tokens + offset for offset in batch_offsets]
            num_tokens += len(batch_embeds)
            all_token_embeds[num_tokens-len(batch_embeds):num_tokens] = batch_embeds

        # Use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher.
        if self.config.index_type ==  XTRIndexType.SCANN:
            assert isinstance(self.config.index_config, XTRScaNNIndexConfig)
            num_neighbors = self.config.index_config.num_neighbors
            max_num_leaves = self.config.index_config.max_num_leaves
            num_leaves_to_search = self.config.index_config.num_leaves_to_search
            max_training_sample_size = self.config.index_config.max_training_sample_size
            dimensions_per_block = self.config.index_config.dimensions_per_block
            anisotropic_quantization_threshold = self.config.index_config.anisotropic_quantization_threshold

            self.searcher = scann.scann_ops_pybind.builder(all_token_embeds[:num_tokens], num_neighbors, "dot_product").tree(
                num_leaves=min(max_num_leaves, num_tokens), num_leaves_to_search=num_leaves_to_search, training_sample_size=min(max_training_sample_size, num_tokens)).score_ah(
                dimensions_per_block, anisotropic_quantization_threshold=anisotropic_quantization_threshold).build()
        elif self.config.index_type ==  XTRIndexType.FAISS:
            assert isinstance(self.config.index_config, XTRFAISSIndexConfig)
            num_clusters = self.config.index_config.num_clusters
            code_size = self.config.index_config.code_size
            nbits_per_idx = self.config.index_config.nbits_per_idx
            opq_matrix_niter = self.config.index_config.opq_matrix_niter

            ds = self.config.token_embed_dim
            quantizer = faiss.IndexFlatIP(ds)
            opq_matrix = faiss.OPQMatrix(ds, code_size)
            opq_matrix.niter = opq_matrix_niter
            sub_index = faiss.IndexIVFPQ(quantizer, ds, num_clusters, code_size, nbits_per_idx, faiss.METRIC_INNER_PRODUCT)
            index = faiss.IndexPreTransform(opq_matrix, sub_index)
            index.train(all_token_embeds[:num_tokens])
            index.add(all_token_embeds[:num_tokens])
            class FaissSearcher(object):
                def __init__(self, index):
                    self.index = index
                def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                    scores, top_ids = self.index.search(query_embeds, final_num_neighbors)
                    return top_ids, scores
            self.searcher = FaissSearcher(index)
        # Used only for small-scale, exact inference.
        elif self.config.index_type == XTRIndexType.BRUTE_FORCE:
            assert isinstance(self.config.index_config, XTRBruteForceIndexConfig)
            self.all_token_embeds = all_token_embeds[:num_tokens]
            class BruteForceSearcher(object):
                def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                    scores = query_embeds.dot(all_token_embeds[:num_tokens].T) # Q x D
                    top_ids = scores.argsort(axis=1)[:, ::-1][:,:final_num_neighbors] # Q x top_k
                    return top_ids, [q_score[q_top_ids] for q_score, q_top_ids in zip(scores, top_ids)] # (Q x top_k, Q x top_k)
            self.searcher = BruteForceSearcher()
        else: raise AssertionError(f"Unsupported XTRIndexType {self.config.index_type}!")

        self.doc_offsets = all_doc_offsets
        self.doc_offsets.append(num_tokens)  # Add final number of tokens.
        self.tid2did = {
            self.doc_offsets[did] + tid: did
            for did in range(len(self.doc_offsets)-1)
            for tid in range(self.doc_offsets[did+1] - self.doc_offsets[did])
        }
        self.tid2did[-1] = 0
        self.docs = documents
        print("Index Ready!", self.searcher)

    def save_index(self):
        if os.path.exists(self.config.path):
            if self.config.override:
                shutil.rmtree(self.config.path)
            else: raise AssertionError(f"Index `{self.config.path}` already exists!")
        os.makedirs(self.config.path, exist_ok=False)

        self._save_pickle(self.config, "config.pickle")
        self._save_json(self.tid2did, "tid2did.json")
        self._save_np(np.array(self.doc_offsets), "doc_offsets.np")
        self._save_pickle(self.docs, "docs.pickle")

        if self.config.index_type ==  XTRIndexType.SCANN:
            assert isinstance(self.config.index_config, XTRScaNNIndexConfig)
            scann_dir = os.path.join(self.config.path, "scann")
            os.makedirs(scann_dir, exist_ok=False)
            self.searcher.serialize(scann_dir)
        elif self.config.index_type ==  XTRIndexType.FAISS:
            assert isinstance(self.config.index_config, XTRFAISSIndexConfig)
            pass
        elif self.config.index_type == XTRIndexType.BRUTE_FORCE:
            assert isinstance(self.config.index_config, XTRBruteForceIndexConfig)
            bruteforce_dir = os.path.join(self.config.path, "bruteforce")
            os.makedirs(bruteforce_dir, exist_ok=False)

            with open(os.path.join(bruteforce_dir, "all_token_embeds.np"), "wb") as file:
                np.save(file, self.all_token_embeds)
        else: raise AssertionError(f"Unsupported XTRIndexType {self.config.index_type}!")

    def _save_pickle(self, data, filename):
        path = os.path.join(self.config.path, filename)
        with open(path, "wb") as file:
            pickle.dump(data, file)

    def _save_json(self, data, filename):
        with open(os.path.join(self.config.path, filename), "w") as file:
            json.dump(data, file)

    def _save_np(self, data, filename):
        with open(os.path.join(self.config.path, filename), "wb") as file:
            np.save(file, data)

    def _batch_search_tokens(self, batch_query, token_top_k, leaves_to_search, pre_reorder_num_neighbors):
        all_query_encodings, query_offsets = self._get_flatten_embeddings(batch_query, return_last_offset=True)
        all_neighbors, all_scores = self.searcher.search_batched(
            all_query_encodings, final_num_neighbors=token_top_k, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors
        )
        return [
            (
                [f'q_{i}' for i in range(query_offsets[oid], query_offsets[oid+1])],  # query_id
                all_neighbors[query_offsets[oid]:query_offsets[oid+1]],  # neighbors
                all_scores[query_offsets[oid]:query_offsets[oid+1]],  # scores
            )
            for oid in range(len(query_offsets)-1)
        ]

    def _estimate_missing_similarity(self, batch_result):
        batch_qtoken_to_ems = [dict() for _ in range(len(batch_result))]
        for b_idx, (query_tokens, _, distances) in enumerate(batch_result):
            for token_idx, qtoken in enumerate(query_tokens):
                idx_t = (token_idx, qtoken)
                # Use similarity of the last token as imputed similarity.
                batch_qtoken_to_ems[b_idx][idx_t] = distances[token_idx][-1]
        return batch_qtoken_to_ems

    def _aggregate_scores(self, batch_result, batch_ems, document_top_k):
        """Aggregates token-level retrieval scores into query-document scores."""

        def get_did2scores(query_tokens, all_neighbors, all_scores):
            did2scores = {}
            # |Q| x k'
            for qtoken_idx, (qtoken, neighbors, scores) in enumerate(zip(query_tokens, all_neighbors, all_scores)):
                for _, (doc_token_id, score) in enumerate(zip(neighbors, scores)):
                    if np.isnan(score):
                        continue
                    docid = self.tid2did[doc_token_id]
                    if docid not in did2scores:
                        did2scores[docid] = {}
                    qtoken_with_idx = (qtoken_idx, qtoken)
                    if qtoken_with_idx not in did2scores[docid]:
                        # Only keep the top score for sum-of-max.
                        did2scores[docid][qtoken_with_idx] = score

            return did2scores
        batch_did2scores = [get_did2scores(qtokens, neighbors, scores) for qtokens, neighbors, scores in batch_result]

        def add_ems(did2scores, query_tokens, ems):
            # |Q| x |Q|k' (assuming most docid is unique)
            for qtoken_idx, qtoken in enumerate(query_tokens):
                qtoken_with_idx = (qtoken_idx, qtoken)
                for _, scores in did2scores.items():
                    if qtoken_with_idx not in scores:
                        scores[qtoken_with_idx] = ems[qtoken_with_idx]
        for did2scores, result, ems in zip(batch_did2scores, batch_result, batch_ems):
            add_ems(did2scores, result[0], ems)

        def get_final_score(did2scores, query_tokens):
            final_qd_score = {}
            # |Q|k' x |Q|
            for docid, scores in did2scores.items():
                assert len(scores) == len(query_tokens)
                final_qd_score[docid] = sum(scores.values()) / len(scores)
            return final_qd_score

        batch_scores = [get_final_score(did2scores, result[0]) for did2scores, result in zip(batch_did2scores, batch_result)]

        batch_ranking = [
            sorted([(docid, score) for docid, score in final_qd_score.items()], key=lambda x: x[1], reverse=True)[:document_top_k]
            for final_qd_score in batch_scores
        ]
        return batch_ranking

    def _get_document_text(self, batch_ranking):
        batch_retrieved_docs = []
        for ranking in batch_ranking:
            retrieved_docs = []
            for did, score in ranking:
                retrieved_docs.append((did, score, self.docs[did]))
            batch_retrieved_docs.append(retrieved_docs)
        return batch_retrieved_docs

    def retrieve_docs(
        self,
        query: Union[str, List[str]],
        token_top_k: Optional[int] = None,
        document_top_k: Optional[int] = None,
        return_text: bool = False,
    ):
        """Runs XTR retrieval for a query."""
        if isinstance(query, List):
            batch_query = query
        else:
            batch_query = [query]
        token_top_k = token_top_k or self.config.token_top_k
        document_top_k = document_top_k or self.config.document_top_k
        if self.config.index_type ==  XTRIndexType.SCANN:
            assert isinstance(self.config.index_config, XTRScaNNIndexConfig)
            leaves_to_search = self.config.index_config.leaves_to_search
            pre_reorder_num_neighbors = self.config.index_config.pre_reorder_num_neighbors
        else:
            leaves_to_search = None
            pre_reorder_num_neighbors = None
        batch_result = self._batch_search_tokens(batch_query, token_top_k=token_top_k, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors)
        batch_mae = self._estimate_missing_similarity(batch_result)
        batch_ranking = self._aggregate_scores(batch_result, batch_mae, document_top_k)
        if return_text:
            return self._get_document_text(batch_ranking), batch_result
        else:
            return batch_ranking, batch_result
