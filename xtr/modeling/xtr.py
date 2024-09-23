import os
import sys
import shutil
import pickle
from typing import Optional
from tqdm import tqdm

import torch
import numpy as np
import tensorflow as tf

import scann
import faiss

from xtr.config import XTRConfig, XTRIndexType, XTRScaNNIndexConfig, XTRFAISSIndexConfig, XTRBruteForceIndexConfig
from xtr.data.collection import Collection
from xtr.data.queries import Queries
from xtr.data.rankings import Rankings
from xtr.modeling.encoder import XTREncoder
from xtr.utils.tracker import NOPTracker

# Extracted from: https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
class BruteForceSearcher(object):
    def __init__(self, all_token_embeds):
        self.all_token_embeds = all_token_embeds

    def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
        scores = query_embeds.numpy().dot(self.all_token_embeds.T) # Q x D
        top_ids = scores.argsort(axis=1)[:, ::-1][:,:final_num_neighbors] # Q x top_k
        return top_ids.copy(), np.array([q_score[q_top_ids] for q_score, q_top_ids in zip(scores, top_ids)]) # (Q x top_k, Q x top_k)

# Extracted from: https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
class FaissSearcher(object):
    def __init__(self, index):
        self.index = index
    def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
        scores, top_ids = self.index.search(query_embeds, final_num_neighbors)
        return top_ids, scores

# Adapted from: https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
class XTR(object):
    def __init__(self, config: XTRConfig, collection: Collection, device=torch.device("cuda")):
        self.config = config

        if not config.is_tpu():
            if device != torch.device("cpu"):
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

        self.encoder = XTREncoder(self.config, device=device)
        if os.path.exists(config.path) and not self.config.override:
            self._load_index(config)
        else:
            self._build_index(Collection.cast(collection))

        if config.index_type == XTRIndexType.FAISS:
            sub_index = faiss.extract_index_ivf(self.searcher.index)
            sub_index.nprobe = 4
        
        self.can_parallel_scann = self.config.index_type == XTRIndexType.SCANN and hasattr(self.searcher.searcher, "set_num_threads")
        self.num_threads = 1

    def _get_token_embeddings(self, texts, maxlen):
        batch_embeds = self.encoder([t.lower() for t in texts], maxlen=maxlen)
        batch_lengths = np.sum(batch_embeds["mask"].cpu().numpy(), axis=1)
        return batch_embeds["encodings"].cpu().numpy(), batch_lengths

    def _get_flatten_embeddings(self, batch_text, maxlen, return_last_offset=False):
        batch_embeddings, batch_lengths = self._get_token_embeddings(batch_text, maxlen=maxlen)
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

    def _build_index(self, collection):
        batch_size = self.config.build_batch_size

        max_num_tokens = len(collection) * self.config.max_seq_len
        # Provide the ability to explicitly set the max number of tokens to prevent OOM issues.
        if self.config.max_num_tokens is not None:
            max_num_tokens = self.config.max_num_tokens
            print(f"#> Explicitly setting max_num_tokens = {max_num_tokens}")

        all_token_embeds = np.zeros((max_num_tokens, self.config.token_embed_dim), dtype=np.float32)
        all_doc_offsets = []
        num_tokens = 0
        for batch_idx, batch_docs in collection.enumerate_batches(batch_size=batch_size):
            batch_embeds, batch_offsets = self._get_flatten_embeddings(batch_docs, maxlen=self.config.doc_maxlen)
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
            self.index = index
            self.searcher = FaissSearcher(index)
        # Used only for small-scale, exact inference.
        elif self.config.index_type == XTRIndexType.BRUTE_FORCE:
            assert isinstance(self.config.index_config, XTRBruteForceIndexConfig)
            self.all_token_embeds = all_token_embeds[:num_tokens]
            self.searcher = BruteForceSearcher(self.all_token_embeds)
        else: raise AssertionError(f"Unsupported XTRIndexType {self.config.index_type}!")

        self.doc_offsets = all_doc_offsets
        self.doc_offsets.append(num_tokens)  # Add final number of tokens.
        self.tid2did = {
            self.doc_offsets[did] + tid: did
            for did in range(len(self.doc_offsets)-1)
            for tid in range(self.doc_offsets[did+1] - self.doc_offsets[did])
        }
        self.tid2did[-1] = 0
        self.collection = collection
        print("Index Ready!", self.searcher)
        self._save_index()

    def _save_index(self):
        print(f"Saving index to {self.config.path}.")
        def save_pickle(data, filename):
            with open(os.path.join(self.config.path, filename), "wb") as file:
                pickle.dump(data, file)

        def save_np(data, filename):
            np.save(os.path.join(self.config.path, filename), data)

        if os.path.exists(self.config.path):
            if self.config.override:
                shutil.rmtree(self.config.path)
            else: raise AssertionError(f"Index `{self.config.path}` already exists!")
        os.makedirs(self.config.path, exist_ok=False)

        save_pickle(self.config, "config.pickle")
        save_pickle(self.tid2did, "tid2did.pickle")
        save_np(np.array(self.doc_offsets), "doc_offsets.npy")
        save_pickle(self.collection, "collection.pickle")

        if self.config.index_type ==  XTRIndexType.SCANN:
            assert isinstance(self.config.index_config, XTRScaNNIndexConfig)
            scann_dir = os.path.join(self.config.path, "scann")
            os.makedirs(scann_dir, exist_ok=False)
            
            self.searcher.serialize(scann_dir)
        elif self.config.index_type ==  XTRIndexType.FAISS:
            assert isinstance(self.config.index_config, XTRFAISSIndexConfig)
            faiss_dir = os.path.join(self.config.path, "faiss")
            os.makedirs(faiss_dir, exist_ok=False)

            faiss.write_index(self.index, os.path.join(faiss_dir, "faiss.index"))
        elif self.config.index_type == XTRIndexType.BRUTE_FORCE:
            assert isinstance(self.config.index_config, XTRBruteForceIndexConfig)
            bruteforce_dir = os.path.join(self.config.path, "bruteforce")
            os.makedirs(bruteforce_dir, exist_ok=False)

            np.save(os.path.join(bruteforce_dir, "all_token_embeds.npy"), self.all_token_embeds)
        else: raise AssertionError(f"Unsupported XTRIndexType {self.config.index_type}!")

    def _load_index(self, load_config: XTRConfig):
        print(f"Loading existing index from {load_config.path}.")
        path = load_config.path
        if not os.path.exists(path):
            raise AssertionError(f"Index does not exist {path}!")
        
        def load_pickle(filename):
            with open(os.path.join(path, filename), "rb") as file:
                return pickle.load(file)

        def load_np(filename):
            return np.load(os.path.join(path, filename))

        config = load_pickle("config.pickle")
        if load_config is not None and not load_config.is_compatible_with(config):
            raise AssertionError(f"Invalid index at {path}!")
        tid2did = load_pickle("tid2did.pickle")
        doc_offsets = load_np("doc_offsets.npy")
        collection = load_pickle("collection.pickle")

        self.config = config
        self.tid2did = tid2did
        self.doc_offsets = doc_offsets
        self.collection = collection

        if config.index_type ==  XTRIndexType.SCANN:
            assert isinstance(config.index_config, XTRScaNNIndexConfig)
            scann_dir = os.path.join(config.path, "scann")
            searcher = scann.scann_ops_pybind.load_searcher(scann_dir)
            self.searcher = searcher
        elif config.index_type ==  XTRIndexType.FAISS:
            assert isinstance(config.index_config, XTRFAISSIndexConfig)
            faiss_dir = os.path.join(config.path, "faiss")
            index = faiss.read_index(os.path.join(faiss_dir, "faiss.index"))
            self.index = index
            self.searcher = FaissSearcher(self.index)
        elif config.index_type == XTRIndexType.BRUTE_FORCE:
            assert isinstance(config.index_config, XTRBruteForceIndexConfig)
            bruteforce_dir = os.path.join(config.path, "bruteforce")
            all_token_embeds = np.load(os.path.join(bruteforce_dir, "all_token_embeds.npy"))
            self.all_token_embeds = all_token_embeds
            self.searcher = BruteForceSearcher(self.all_token_embeds)
        else: raise AssertionError(f"Unsupported XTRIndexType {config.index_type}!")

    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
        if self.config.index_type ==  XTRIndexType.SCANN:
            if self.can_parallel_scann:
                self.searcher.searcher.set_num_threads(num_threads)
            else: print("#> [WARNING] Cannot set_num_threads for ScaNN", file=sys.stderr)

    def _searcher_search_batched(self, all_query_encodings, final_num_neighbors, leaves_to_search, pre_reorder_num_neighbors):
        if self.can_parallel_scann and self.num_threads != 1:
            return self.searcher.search_batched_parallel(
                all_query_encodings, final_num_neighbors=final_num_neighbors, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors
            )
        return self.searcher.search_batched(
            all_query_encodings, final_num_neighbors=final_num_neighbors, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors
        )

    def _batch_search_tokens(self, batch_query, token_top_k, leaves_to_search, pre_reorder_num_neighbors, tracker):
        tracker.begin("Query Encoding")
        all_query_encodings, query_offsets = self._get_flatten_embeddings(batch_query, maxlen=self.config.query_maxlen, return_last_offset=True)
        tracker.end("Query Encoding")

        tracker.begin("search_batched")
        all_neighbors, all_scores = self._searcher_search_batched(
            all_query_encodings, final_num_neighbors=token_top_k, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors
        )
        tracker.end("search_batched")

        tracker.begin("enumerate_scores")
        result = [
            (
                [f'q_{i}' for i in range(query_offsets[oid], query_offsets[oid+1])],  # query_id
                all_neighbors[query_offsets[oid]:query_offsets[oid+1]],  # neighbors
                all_scores[query_offsets[oid]:query_offsets[oid+1]],  # scores
            )
            for oid in range(len(query_offsets)-1)
        ]
        tracker.end("enumerate_scores")
        return result

    def _estimate_missing_similarity(self, batch_result, tracker):
        tracker.begin("Estimate Missing Similarity")
        batch_qtoken_to_ems = [dict() for _ in range(len(batch_result))]
        for b_idx, (query_tokens, _, distances) in enumerate(batch_result):
            for token_idx, qtoken in enumerate(query_tokens):
                idx_t = (token_idx, qtoken)
                # Use similarity of the last token as imputed similarity.
                batch_qtoken_to_ems[b_idx][idx_t] = distances[token_idx][-1]
        tracker.end("Estimate Missing Similarity")
        return batch_qtoken_to_ems

    def _aggregate_scores(self, batch_result, batch_ems, document_top_k, tracker):
        """Aggregates token-level retrieval scores into query-document scores."""

        tracker.begin("get_did2scores")
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
        tracker.end("get_did2scores")

        tracker.begin("add_ems")
        def add_ems(did2scores, query_tokens, ems):
            # |Q| x |Q|k' (assuming most docid is unique)
            for qtoken_idx, qtoken in enumerate(query_tokens):
                qtoken_with_idx = (qtoken_idx, qtoken)
                for _, scores in did2scores.items():
                    if qtoken_with_idx not in scores:
                        scores[qtoken_with_idx] = ems[qtoken_with_idx]

        for did2scores, result, ems in zip(batch_did2scores, batch_result, batch_ems):
            add_ems(did2scores, result[0], ems)
        tracker.end("add_ems")

        tracker.begin("get_final_score")
        def get_final_score(did2scores, query_tokens):
            final_qd_score = {}
            # |Q|k' x |Q|
            for docid, scores in did2scores.items():
                assert len(scores) == len(query_tokens)
                final_qd_score[docid] = sum(scores.values()) / len(scores)
            return final_qd_score

        batch_scores = [get_final_score(did2scores, result[0]) for did2scores, result in zip(batch_did2scores, batch_result)]
        tracker.end("get_final_score")

        tracker.begin("sort_scores")
        batch_ranking = [
            sorted([(docid, score) for docid, score in final_qd_score.items()], key=lambda x: x[1], reverse=True)[:document_top_k]
            for final_qd_score in batch_scores
        ]
        tracker.end("sort_scores")
        return batch_ranking

    def _get_document_text(self, batch_ranking):
        batch_retrieved_docs = []
        for ranking in batch_ranking:
            retrieved_docs = []
            for did, score in ranking:
                retrieved_docs.append((did, score, self.collection[did]))
            batch_retrieved_docs.append(retrieved_docs)
        return batch_retrieved_docs

    def retrieve_docs(
        self,
        query,
        token_top_k: Optional[int] = None,
        document_top_k: Optional[int] = None,
        return_text: bool = False,
        tracker = NOPTracker()
    ):
        """Runs XTR retrieval for a query."""
        rankings = dict()

        batch_query = Queries.cast(query)
        token_top_k = token_top_k or self.config.token_top_k
        document_top_k = document_top_k or self.config.document_top_k
        for query_id, query_text in tqdm(batch_query):
            tracker.next_iteration()
            if self.config.index_type ==  XTRIndexType.SCANN:
                assert isinstance(self.config.index_config, XTRScaNNIndexConfig)
                leaves_to_search = self.config.index_config.leaves_to_search
                pre_reorder_num_neighbors = self.config.index_config.pre_reorder_num_neighbors
            else:
                leaves_to_search = None
                pre_reorder_num_neighbors = None
            batch_result = self._batch_search_tokens([query_text], token_top_k=token_top_k, leaves_to_search=leaves_to_search,
                                                     pre_reorder_num_neighbors=pre_reorder_num_neighbors, tracker=tracker)
            batch_mae = self._estimate_missing_similarity(batch_result, tracker=tracker)
            batch_ranking = self._aggregate_scores(batch_result, batch_mae, document_top_k, tracker=tracker)
            
            # TODO(jlscheerer) Fix this for multiple queries
            results = self._get_document_text(batch_ranking) if return_text else batch_ranking
            rankings[query_id] = results[0]
            tracker.end_iteration()

        return Rankings(data=rankings, index_map=self.collection.index_map(), text=return_text)
