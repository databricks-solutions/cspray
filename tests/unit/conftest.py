import pytest
import tempfile
import os

import pandas as pd
import numpy as np

@pytest.fixture(scope="module")
def dummy_h5ad_pair():
    """Two tiny hand-built h5ad files with deliberately different obs schemas.

    Known values let tests assert exact per-row metadata assignment, coverage,
    action classification, per-file localisation, and promotion. Two files (not
    one) are needed because within a single file every row has every column, so
    partial coverage / per-file / alias / type-conflict only arise across files.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    # File A: 3 cells x 4 genes (X must be CSR for cspray's format check)
    A = ad.AnnData(
        X=csr_matrix(np.arange(12, dtype="float32").reshape(3, 4)),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(["T", "B", "T"]),  # categorical path
            "tissue":     ["lung", "lung", "lung"],         # sample-level, differs by file
            "donor_id":  ["dA0", "dA1", "dA2"],            # unique -> id-like
            "organism":  ["human", "human", "human"],      # constant
            "n_genes":   [10, 20, 30],
            "qc_score":  [0.1, 0.2, 0.3],
            "CellType":  ["T", "B", "T"],                  # alias of cell_type, file A only
            "mixed":     [1, 2, 3],                        # int here...
        }, index=["cA0", "cA1", "cA2"]),
        var=pd.DataFrame({
            "feature_type": ["gene"] * 4,
            "highly_variable": [True, False, True, False],
        }, index=["g0", "g1", "g2", "g3"]),
    )

    # File B: 3 cells, different obs schema
    B = ad.AnnData(
        X=csr_matrix(np.arange(12, 24, dtype="float32").reshape(3, 4)),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(["T", "B", "B"]),
            "tissue":     ["blood", "blood", "blood"],
            "donor_id":  ["dB0", "dB1", "dB2"],
            "organism":  ["human", "human", "human"],
            "n_genes":   [15, 25, 35],
            "qc_score":  [0.15, 0.25, 0.35],
            "batch":     ["b1", "b1", "b2"],               # file B only -> 50% coverage
            "mixed":     ["x", "y", "z"],                  # ...str here -> type conflict
        }, index=["cB0", "cB1", "cB2"]),
        var=pd.DataFrame({
            "feature_type": ["gene"] * 4,
            "highly_variable": [True, True, False, False],
        }, index=["g0", "g1", "g2", "g3"]),
    )

    fa = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    fb = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    A.write(fa.name)
    B.write(fb.name)

    yield [fa.name, fb.name]

    os.remove(fa.name)
    os.remove(fb.name)

@pytest.fixture(scope="module")
def dummy_h5ad_categorical_index():
    """A tiny h5ad whose obs index is stored as a categorical (a group on disk).

    Some real-world files set the obs index from a categorical column, so the
    `_index` element becomes an h5py group (codes/categories) rather than a flat
    dataset. This exercises the row-count path (`udf_get_n_obs`) which must fall
    back to the codes length instead of assuming a `.shape`.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["T", "B", "T"]),
            "n_genes": [10, 20, 30],
        },
        index=pd.CategoricalIndex(["cA0", "cA1", "cA2"], name="barcode"),
    )
    adata = ad.AnnData(
        X=csr_matrix(np.arange(1, 13, dtype="float32").reshape(3, 4)),
        obs=obs,
        var=pd.DataFrame(
            {"feature_type": ["gene"] * 4},
            index=["g0", "g1", "g2", "g3"],
        ),
    )

    f = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    adata.write(f.name)

    yield f.name

    os.remove(f.name)

@pytest.fixture(scope="module")
def dummy_h5ad_dense():
    """Dense (non-CSR) AnnData. Format check must reject it."""
    import anndata as ad

    adata = ad.AnnData(
        X=np.arange(12, dtype="float32").reshape(3, 4),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    adata.write(f.name)
    yield f.name
    os.remove(f.name)

@pytest.fixture(scope="module")
def dummy_h5ad_empty_x():
    """A CSR h5ad with cells/genes but zero stored values (nnz == 0).

    Guards the expression-chunk path: sequence stop is nnz-1, so nnz == 0
    must not emit a chunk (and must not crash).
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    adata = ad.AnnData(
        X=csr_matrix((3, 4), dtype="float32"),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    adata.write(f.name)
    yield f.name
    os.remove(f.name)
