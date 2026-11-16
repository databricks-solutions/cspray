import json
import pytest

from cspray.data import SprayData
from cspray.read import (
    _explode_nnz_ranges,
    assign_fp_int,
    construct_h5ad_path_df,
    inspect_h5ads,
)
from pyspark.sql import functions as F

@pytest.mark.parametrize("nnz,chunk_size", [
    (0, 4),
    (12, 1),
    (12, 4),
    (12, 5),
    (12, 12),
    (30_000_000, 30_000_000),
])
def test_explode_nnz_ranges_never_starts_at_nnz(spark_collect, nnz, chunk_size):
    """sequence stop must be nnz-1 so a multiple of chunk_size does not emit
    start_idx == nnz (that chunk IndexErrors in the CSR slice).
    """
    sdf = spark_collect.createDataFrame([(nnz,)], ["maxsize"])
    rows = _explode_nnz_ranges(sdf, "maxsize", chunk_size, None).collect()
    if nnz == 0:
        assert rows == []
        return
    starts = [r.start_idx for r in rows]
    ends = [r.end_idx for r in rows]
    assert min(starts) == 0
    assert max(ends) == nnz - 1
    assert all(s < nnz for s in starts)
    assert all(s <= e for s, e in zip(starts, ends))
    if nnz <= 12:
        covered = [i for s, e in zip(starts, ends) for i in range(s, e + 1)]
        assert covered == list(range(nnz))

@pytest.mark.parametrize("chunk_size", [1, 4, 5, 12])
def test_expression_chunk_boundary(dummy_h5ad_categorical_index, spark_collect, chunk_size):
    """3x4 dense CSR → nnz=12. chunk_size 1/4/12 divide nnz exactly and used
    to emit a start_idx==nnz chunk. 5 does not divide 12 (control).
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_categorical_index, from_raw=False,
        mode='delta', force_partitioning=1, chunk_size=chunk_size,
    )
    assert sdata.X.count() == 12
    assert sdata.obs.count() == 3
    assert sdata.var.count() == 4

def test_assign_fp_int_none_matches_empty_registry(spark_collect):
    listing = spark_collect.createDataFrame(
        [('/tmp/b.h5ad',), ('/tmp/a.h5ad',)], ['file_path']
    )
    empty = spark_collect.createDataFrame([], 'file_path string, fp_int int')
    from_none = {r.file_path: r.fp_int for r in assign_fp_int(listing).collect()}
    from_empty = {r.file_path: r.fp_int for r in assign_fp_int(listing, empty).collect()}
    assert from_none == from_empty == {'/tmp/a.h5ad': 0, '/tmp/b.h5ad': 1}


def test_assign_fp_int_keeps_hits_and_increments(spark_collect):
    listing = spark_collect.createDataFrame(
        [('/tmp/a.h5ad',), ('/tmp/b.h5ad',), ('/tmp/c.h5ad',)],
        ['file_path'],
    )
    existing = spark_collect.createDataFrame(
        [('/tmp/a.h5ad', 7)], ['file_path', 'fp_int']
    )
    out = {r.file_path: r.fp_int for r in assign_fp_int(listing, existing).collect()}
    assert out['/tmp/a.h5ad'] == 7
    assert out['/tmp/b.h5ad'] == 8
    assert out['/tmp/c.h5ad'] == 9


def test_assign_fp_int_never_renumbers(spark_collect):
    listing = spark_collect.createDataFrame([('/tmp/a.h5ad',)], ['file_path'])
    existing = spark_collect.createDataFrame(
        [('/tmp/a.h5ad', 7)], ['file_path', 'fp_int']
    )
    again = assign_fp_int(listing, existing).collect()[0].fp_int
    assert again == 7


def test_from_h5ads_existing_sam_keeps_fp_int(dummy_h5ad_pair, spark_collect):
    a, b = sorted(dummy_h5ad_pair)
    existing = spark_collect.createDataFrame([(a, 7)], ['file_path', 'fp_int'])
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2, existing_sam=existing,
    )
    by_path = {r.file_path: r.fp_int for r in sdata.sam.collect()}
    assert by_path[a] == 7
    assert by_path[b] == 8
    assert {r.fp_int for r in sdata.X.select('fp_int').distinct().collect()} == {7, 8}

def test_injected_fp_int_used_everywhere(dummy_h5ad_categorical_index, spark_collect):
    """If the listing already has fp_int, do not re-hash; X/obs/var/sam all use it."""
    listing = construct_h5ad_path_df(
        path=dummy_h5ad_categorical_index, spark=spark_collect
    ).withColumn('fp_int', F.lit(42).cast('int'))
    sdata = SprayData.from_h5ads(
        spark_collect, df=listing, from_raw=False,
        mode='delta', force_partitioning=1,
    )
    assert sdata.sam.collect()[0].fp_int == 42
    assert {r.fp_int for r in sdata.X.select('fp_int').distinct().collect()} == {42}
    assert {r.fp_int for r in sdata.obs.select('fp_int').distinct().collect()} == {42}
    assert {r.fp_int for r in sdata.var.select('fp_int').distinct().collect()} == {42}

def test_inspect_maxsize_matches_h5py(dummy_h5ad_categorical_index, spark_collect):
    """Inspect nnz equals the on-disk CSR data length (categorical dummy nnz=12)."""
    import h5py
    listing = construct_h5ad_path_df(
        path=dummy_h5ad_categorical_index, spark=spark_collect
    )
    row = inspect_h5ads(listing, spark_collect, force_partitioning=1).collect()[0]
    with h5py.File(dummy_h5ad_categorical_index, 'r') as f:
        assert row.maxsize_main == f['X']['data'].shape[0]
        assert row.dtype_main == str(f['X']['data'].dtype)
    assert row.maxsize_main == 12
    assert row.main_csr
    assert not row.has_raw

def test_inspect_empty_x_maxsize(dummy_h5ad_empty_x, spark_collect):
    listing = construct_h5ad_path_df(path=dummy_h5ad_empty_x, spark=spark_collect)
    row = inspect_h5ads(listing, spark_collect, force_partitioning=1).collect()[0]
    assert row.maxsize_main == 0
    assert row.main_csr

def test_inspect_dtype_present(dummy_h5ad_categorical_index, spark_collect):
    listing = construct_h5ad_path_df(
        path=dummy_h5ad_categorical_index, spark=spark_collect
    )
    row = inspect_h5ads(listing, spark_collect, force_partitioning=1).collect()[0]
    assert row.dtype_main == 'float32'

def test_dense_h5ad_rejected(dummy_h5ad_dense, spark_collect):
    with pytest.raises(ValueError, match='default X with a CSR format'):
        SprayData.from_h5ads(
            spark_collect, path=dummy_h5ad_dense, from_raw=False,
            mode='delta', force_partitioning=1,
        )

def test_expression_empty_x(dummy_h5ad_empty_x, spark_collect):
    """nnz == 0 must ingest obs/var and produce an empty X, not IndexError."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_empty_x, from_raw=False,
        mode='delta', force_partitioning=1, chunk_size=4,
    )
    assert sdata.X.count() == 0
    assert sdata.obs.count() == 3
    assert sdata.var.count() == 4

def test_categorical_index_row_count(dummy_h5ad_categorical_index, spark_collect):
    """Regression: obs whose index is categorical is stored as an h5py group, so
    the row-count path must use the codes length rather than a `.shape` on the
    index element. Reads a file with a categorical obs index end-to-end.
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_categorical_index, from_raw=False,
        mode='delta', force_partitioning=1,
    )

    # all three cells are read despite the categorical (group-encoded) index
    assert sdata.obs.count() == 3
    assert sdata.var.count() == 4

    cell_idx = sdata.obs.select('cell_idx').toPandas()['cell_idx'].values
    assert set(cell_idx) == {0, 1, 2}

def test_categorical_chunked_decode(dummy_h5ad_categorical_index, spark_collect):
    """Categoricals are decoded per-chunk by reading only the referenced
    categories and renumbering the codes. With metadata_chunk_size=2 the 3-row
    file splits across a chunk boundary (rows 0-1, then row 2), so different
    chunks see different category subsets; the decoded values must still be
    correct. Guards the bounded-categorical-read path.
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_categorical_index, from_raw=False,
        mode='delta', force_partitioning=1,
        obs_metadata_columns='all', metadata_variant=False,
        metadata_chunk_size=2,
    )

    obs = sdata.obs.select('cell_barcode', 'obs_data').toPandas()
    # categorical index barcodes decode correctly across the chunk boundary
    assert set(obs['cell_barcode']) == {'cA0', 'cA1', 'cA2'}
    parsed = {r.cell_barcode: json.loads(r.obs_data) for r in obs.itertuples()}
    # cell_type is categorical; row 2 (cA2) is in a chunk that only references 'T'
    assert parsed['cA0']['cell_type'] == 'T'
    assert parsed['cA1']['cell_type'] == 'B'
    assert parsed['cA2']['cell_type'] == 'T'
    assert parsed['cA0']['n_genes'] == 10
    assert parsed['cA2']['n_genes'] == 30
