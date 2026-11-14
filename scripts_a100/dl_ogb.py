import os, numpy as np, scipy.sparse as sp, torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})
import ogb.utils.url
ogb.utils.url.decide_download = lambda url: True
from ogb.nodeproppred import NodePropPredDataset
from scipy.io import mmwrite
OUT="/dev/shm/real_matrices"; os.makedirs(OUT, exist_ok=True)
for name, tag in [("ogbn-arxiv","ogbn_arxiv"),("ogbn-products","ogbn_products")]:
    d = NodePropPredDataset(name=name, root="/dev/shm/ogb")
    graph, _ = d[0]
    ei = np.asarray(graph["edge_index"], dtype=np.int64)
    n = int(graph["num_nodes"])
    print(name, "nodes", n, "edges_raw", ei.shape[1], flush=True)
    A = sp.coo_matrix((np.ones(ei.shape[1], np.int32), (ei[0], ei[1])), shape=(n,n)).tocsr()
    A.sum_duplicates(); A.sort_indices()
    A.data = np.ones(A.nnz, np.int32)
    print(name, "csr nnz", A.nnz, flush=True)
    p = os.path.join(OUT, tag, tag + ".mtx"); os.makedirs(os.path.dirname(p), exist_ok=True)
    mmwrite(p, sp.coo_matrix(A), field="integer")
    print("written", p, flush=True)
