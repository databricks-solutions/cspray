import os
import pytest

from cspray.data import SAM_REGISTRY_COLUMNS, SprayData, sam_from_listing
from cspray.read import construct_h5ad_path_df
from pyspark.sql import functions as F
import cspray as cs

def _assert_listing_matches_stat(row, path):
    st = os.stat(path)
    assert row.file_path == path
    assert row.source_length == st.st_size
    assert abs(row.source_modified - st.st_mtime) < 1e-3

def test_listing_single_file(dummy_h5ad_categorical_index, spark_collect):
    sdf = construct_h5ad_path_df(path=dummy_h5ad_categorical_index, spark=spark_collect)
    assert set(sdf.columns) == {'file_path', 'source_length', 'source_modified'}
    rows = sdf.collect()
    assert len(rows) == 1
    _assert_listing_matches_stat(rows[0], dummy_h5ad_categorical_index)

def test_listing_path_list(dummy_h5ad_pair, spark_collect):
    sdf = construct_h5ad_path_df(path=dummy_h5ad_pair, spark=spark_collect)
    got = {r.file_path: r for r in sdf.collect()}
    assert set(got) == set(dummy_h5ad_pair)
    for p in dummy_h5ad_pair:
        _assert_listing_matches_stat(got[p], p)

def test_listing_directory(dummy_h5ad_pair, spark_collect, tmp_path):
    for src in dummy_h5ad_pair:
        os.symlink(src, tmp_path / os.path.basename(src))
    (tmp_path / 'ignore.txt').write_text('not an h5ad')
    sdf = construct_h5ad_path_df(path=str(tmp_path), spark=spark_collect)
    rows = sdf.collect()
    assert len(rows) == 2
    got_names = {os.path.basename(r.file_path) for r in rows}
    assert got_names == {os.path.basename(p) for p in dummy_h5ad_pair}
    for r in rows:
        _assert_listing_matches_stat(r, r.file_path)

def test_listing_df_passthrough(dummy_h5ad_categorical_index, spark_collect):
    incoming = spark_collect.createDataFrame(
        [(dummy_h5ad_categorical_index,)], ['file_path']
    )
    out = construct_h5ad_path_df(df=incoming, spark=spark_collect)
    assert out.columns == incoming.columns
    assert [r.file_path for r in out.collect()] == [dummy_h5ad_categorical_index]

def test_listing_path_and_df_rejected(dummy_h5ad_categorical_index, spark_collect):
    incoming = spark_collect.createDataFrame(
        [(dummy_h5ad_categorical_index,)], ['file_path']
    )
    with pytest.raises(ValueError, match='only path or df'):
        construct_h5ad_path_df(
            path=dummy_h5ad_categorical_index, df=incoming, spark=spark_collect
        )

def test_sam_from_listing_not_obs(dummy_h5ad_pair, spark_collect):
    """sam is one row per file from the listing, not distinct obs cells."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    assert set(sdata.sam.columns) == set(SAM_REGISTRY_COLUMNS)
    sam = sdata.sam.toPandas()
    assert len(sam) == 2
    assert set(sam['file_path']) == set(dummy_h5ad_pair)
    assert sdata.obs.count() == 6
    x_ids = {r.fp_int for r in sdata.X.select('fp_int').distinct().collect()}
    assert x_ids == set(sam['fp_int'])
    by_path = {r.file_path: r for r in sdata.sam.collect()}
    for p in dummy_h5ad_pair:
        _assert_listing_matches_stat(by_path[p], p)
        assert by_path[p].ingested_at is not None
    ordered = sorted(dummy_h5ad_pair)
    assert by_path[ordered[0]].fp_int == 0
    assert by_path[ordered[1]].fp_int == 1


def test_sam_from_listing_helper_copies_signature(dummy_h5ad_categorical_index, spark_collect):
    listing = construct_h5ad_path_df(
        path=dummy_h5ad_categorical_index, spark=spark_collect
    )
    listing = listing.withColumn('fp_int', F.lit(7).cast('int'))
    sam = sam_from_listing(listing)
    assert set(sam.columns) == set(SAM_REGISTRY_COLUMNS)
    row = sam.collect()[0]
    assert row.fp_int == 7
    _assert_listing_matches_stat(row, dummy_h5ad_categorical_index)
    assert row.ingested_at is not None


def test_sam_from_path_only_df_has_null_signature(dummy_h5ad_categorical_index, spark_collect):
    incoming = spark_collect.createDataFrame(
        [(dummy_h5ad_categorical_index,)], ['file_path']
    )
    sam = sam_from_listing(incoming)
    row = sam.collect()[0]
    assert row.file_path == dummy_h5ad_categorical_index
    assert row.source_length is None
    assert row.source_modified is None
    assert row.ingested_at is not None
    assert row.fp_int == 0


def test_sam_registry_columns_not_overwritten(dummy_h5ad_pair, spark_collect):
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    before = (
        sdata.sam.select('file_path', 'source_length').toPandas()
        .sort_values('file_path').reset_index(drop=True)
    )
    poison = spark_collect.createDataFrame(
        [(dummy_h5ad_pair[0], 1), (dummy_h5ad_pair[1], 2)],
        ['file_path', 'source_length'],
    )
    with pytest.raises(ValueError, match='registry columns'):
        cs.md.add_sample_metadata(sdata, poison)
    after = (
        sdata.sam.select('file_path', 'source_length').toPandas()
        .sort_values('file_path').reset_index(drop=True)
    )
    assert after.equals(before)
