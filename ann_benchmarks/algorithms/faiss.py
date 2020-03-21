from __future__ import absolute_import

import ctypes
import sys

import faiss
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN

sys.path.append("install/lib-faiss")  # noqa


class Faiss(BaseANN):
    def query(self, v, n):
        if self._metric == "angular":
            v /= numpy.linalg.norm(v)
        D, I = self.index.search(numpy.expand_dims(v, axis=0).astype(numpy.float32), n)
        return I[0]

    def batch_query(self, X, n):
        if self._metric == "angular":
            X /= numpy.linalg.norm(X)
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res


class FaissFlat(Faiss):
    """Flat Index:

    Flat indexes just encode the vectors into codes of a fixed size and
    store them in an array of ntotal * code_size bytes.

    At search time, all the indexed vectors are decoded sequentially and
    compared to the query vectors.
    For the IndexPQ the comparison is done in the compressed domain, which is faster.

    The available encodings are (from least to strongest compression):

        * IndexFlat: No encoding at all.
            The vectors are stored without compression;
        * 16-bit float encoding (IndexScalarQuantizer with QT_fp16):
            the vectors are compressed to 16-bit floats, which may cause some loss of precision;
        * IndexScalarQuantizer with QT_8bit/QT_6bit/QT_4bit: 8/6/4-bit integer encoding
            (vectors quantized to 256/64/16 levels)
        * IndexPQ: PQ encoding.
            Vectors are split into sub-vectors that are each quantized to a few bits (usually 8).


    """

    def __init__(self, n_dims, encoding=None):
        self.n_dims = n_dims
        self.index = None
        self.name = f"FaissFlatL2(n_dims={self.n_dims})"

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        f = X.shape[1]
        self.index = faiss.IndexFlatL2(f)
        self.index.train(X)
        self.index.add(X)


class FaissLSH(Faiss):
    def __init__(self, metric, n_bits):
        self._n_bits = n_bits
        self.index = None
        self._metric = metric
        self.name = "FaissLSH(n_bits={})".format(self._n_bits)

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        f = X.shape[1]
        self.index = faiss.IndexLSH(f, self._n_bits)
        self.index.train(X)
        self.index.add(X)


class FaissIVF(Faiss):

    """Inverse File Index: Non-exhaustive searches.

    The IndexIVF class (and its children) is used for large-scale indices.
    It clusters all input vectors into nlist groups (nlist is a field of IndexIVF).
    At add time, a vector is assigned to a groups. At search time, the most similar
    groups to the query vector are identified and scanned exhaustively.

    Thus, the IndexIVF has two components:

        * Quantizer (aka coarse quantizer) index:
        Given a vector, the search function of the quantizer index
        returns the group the vector belongs to.
        When searched with nprobe>1 results, it returns the nprobe
        nearest groups to the query vector (nprobe is a field of IndexIVF).

        * InvertedLists object:
        This object maps a group id (in 0..nlist-1), to a sequence of (code, id) pairs.

    """

    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if self._metric == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], self._n_list, faiss.METRIC_L2
        )
        index.train(X)
        index.add(X)
        self.index = index

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {
            "dist_comps": faiss.cvar.indexIVF_stats.ndis
            + faiss.cvar.indexIVF_stats.nq * self._n_list  # noqa
        }

    def __str__(self):
        return "FaissIVF(n_list=%d, n_probe=%d)" % (self._n_list, self._n_probe)
