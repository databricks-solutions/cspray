import os
import shutil
import pytest
import pandas as pd
import numpy as np

from cspray.data import SprayData
from cspray import ingest
from pyspark.sql import functions as F

def test_ingest_diff_classifies(spark_collect):
    listing = spark_collect.createDataFrame(
        [('/a.h5ad', 10, 1.0), ('/b.h5ad', 20, 2.0), ('/c.h5ad', 30, 3.0)],
        ['file_path', 'source_length', 'source_modified'],
    )
    sam = spark_collect.createDataFrame(
        [
            (0, '/a.h5ad', 10, 1.0),
            (1, '/b.h5ad', 99, 2.0),
            (2, '/gone.h5ad', 1, 0.0),
        ],
        ['fp_int', 'file_path', 'source_length', 'source_modified'],
    )
    batch, stats = ingest.diff(listing, sam)
    assert {r.file_path for r in batch.collect()} == {'/b.h5ad', '/c.h5ad'}
    assert stats == {'n_new': 1, 'n_changed': 1, 'n_skipped': 1, 'n_forced': 0}


def test_ingest_diff_force_path_and_fp_int(spark_collect):
    listing = spark_collect.createDataFrame(
        [('/a.h5ad', 10, 1.0), ('/b.h5ad', 20, 2.0)],
        ['file_path', 'source_length', 'source_modified'],
    )
    sam = spark_collect.createDataFrame(
        [(0, '/a.h5ad', 10, 1.0), (1, '/b.h5ad', 20, 2.0)],
        ['fp_int', 'file_path', 'source_length', 'source_modified'],
    )
    batch, stats = ingest.diff(listing, sam, force=['/a.h5ad'])
    assert {r.file_path for r in batch.collect()} == {'/a.h5ad'}
    assert stats['n_forced'] == 1
    assert stats['n_skipped'] == 1
    batch, stats = ingest.diff(listing, sam, force=[1])
    assert {r.file_path for r in batch.collect()} == {'/b.h5ad'}
    assert stats['n_forced'] == 1


def test_ingest_diff_no_sam_is_all_new(spark_collect):
    listing = spark_collect.createDataFrame(
        [('/a.h5ad', 10, 1.0)],
        ['file_path', 'source_length', 'source_modified'],
    )
    batch, stats = ingest.diff(listing, None)
    assert batch.count() == 1
    assert stats['n_new'] == 1
    assert stats['n_skipped'] == 0


def test_ingest_run_rejects_unimplemented_flags():
    with pytest.raises(ValueError, match='if_changed'):
        ingest.run(None, if_changed='skip')
    with pytest.raises(ValueError, match='if_removed'):
        ingest.run(None, if_removed='drop')


def _link_pair(dummy_h5ad_pair, dest):
    dest.mkdir()
    paths = []
    for src in dummy_h5ad_pair:
        p = dest / os.path.basename(src)
        os.symlink(src, p)
        paths.append(str(p))
    return paths


def _run_kw(spark, path, tables):
    return dict(
        spark=spark, path=path,
        table_base=tables, join_char='/', mode='delta',
        from_raw=False, force_partitioning=2,
        create_if_not_exists=True,
    )


def test_ingest_run_missing_sam_requires_flag(spark_collect, tmp_path):
    with pytest.raises(ValueError, match='create_if_not_exists'):
        ingest.run(
            spark_collect,
            path=str(tmp_path),
            table_base=str(tmp_path / 'tables'),
            join_char='/',
            mode='delta',
        )


def test_ingest_run_x_without_sam_refuses_overwrite(
    dummy_h5ad_pair, spark_collect, tmp_path
):
    paths = _link_pair(dummy_h5ad_pair, tmp_path / 'h5ads')
    tables = str(tmp_path / 'tables')
    sdata = SprayData.from_h5ads(
        spark_collect, path=paths, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    sdata.to_tables(
        spark_collect, table_base=tables, join_char='/', subset=['X']
    )
    with pytest.raises(ValueError, match='exists'):
        ingest.run(**_run_kw(spark_collect, str(tmp_path / 'h5ads'), tables))


def test_ingest_run_second_pass_skips(dummy_h5ad_pair, spark_collect, tmp_path):
    paths = _link_pair(dummy_h5ad_pair, tmp_path / 'h5ads')
    tables = str(tmp_path / 'tables')
    kw = _run_kw(spark_collect, str(tmp_path / 'h5ads'), tables)
    r1 = ingest.run(**kw)
    assert r1['n_new'] == 2 and r1['wrote']
    sam1 = {
        r.file_path: r.ingested_at
        for r in spark_collect.read.format('delta').load(tables + '/sam').collect()
    }
    r2 = ingest.run(**kw)
    assert r2['n_skipped'] == 2 and not r2['wrote']
    sam2 = {
        r.file_path: r.ingested_at
        for r in spark_collect.read.format('delta').load(tables + '/sam').collect()
    }
    assert sam1 == sam2
    assert set(sam1) == set(paths)


def test_ingest_run_adds_new_file(dummy_h5ad_pair, spark_collect, tmp_path):
    h5dir = tmp_path / 'h5ads'
    h5dir.mkdir()
    a, b = dummy_h5ad_pair
    os.symlink(a, h5dir / 'a.h5ad')
    tables = str(tmp_path / 'tables')
    kw = _run_kw(spark_collect, str(h5dir), tables)
    r1 = ingest.run(**kw)
    assert r1['n_new'] == 1
    os.symlink(b, h5dir / 'b.h5ad')
    r2 = ingest.run(**kw)
    assert r2['n_new'] == 1 and r2['n_skipped'] == 1 and r2['wrote']
    sam = spark_collect.read.format('delta').load(tables + '/sam')
    assert sam.count() == 2
    ids = {r.fp_int for r in sam.collect()}
    assert ids == {0, 1}
    x_ids = {
        r.fp_int
        for r in spark_collect.read.format('delta').load(tables + '/X')
        .select('fp_int').distinct().collect()
    }
    assert x_ids == {0, 1}


def test_ingest_run_changed_file_keeps_other(dummy_h5ad_pair, spark_collect, tmp_path):
    import anndata as ad
    from scipy.sparse import csr_matrix

    h5dir = tmp_path / 'h5ads'
    h5dir.mkdir()
    a = h5dir / 'a.h5ad'
    b = h5dir / 'b.h5ad'
    shutil.copy(dummy_h5ad_pair[0], a)
    shutil.copy(dummy_h5ad_pair[1], b)
    tables = str(tmp_path / 'tables')
    kw = _run_kw(spark_collect, str(h5dir), tables)
    ingest.run(**kw)
    ad.AnnData(
        X=csr_matrix(np.ones((3, 4), dtype='float32')),
        obs=pd.DataFrame(index=['c0', 'c1', 'c2']),
        var=pd.DataFrame(index=['g0', 'g1', 'g2', 'g3']),
    ).write(a)
    r2 = ingest.run(**kw)
    assert r2['n_changed'] == 1 and r2['n_skipped'] == 1
    X = spark_collect.read.format('delta').load(tables + '/X')
    a_id = (
        spark_collect.read.format('delta').load(tables + '/sam')
        .filter(F.col('file_path').endswith('a.h5ad'))
        .collect()[0].fp_int
    )
    b_id = 1 - a_id
    a_expr = {
        r.expression
        for r in X.filter(F.col('fp_int') == a_id).select('expression').distinct().collect()
    }
    assert a_expr == {1.0}
    b_expr = {
        r.expression
        for r in X.filter(F.col('fp_int') == b_id).select('expression').distinct().collect()
    }
    assert 1.0 not in b_expr
    assert X.select('fp_int').distinct().count() == 2


def test_ingest_run_removed_file_kept(dummy_h5ad_pair, spark_collect, tmp_path):
    h5dir = tmp_path / 'h5ads'
    paths = _link_pair(dummy_h5ad_pair, h5dir)
    tables = str(tmp_path / 'tables')
    kw = _run_kw(spark_collect, str(h5dir), tables)
    ingest.run(**kw)
    os.unlink(paths[0])
    r2 = ingest.run(**kw)
    assert r2['n_skipped'] == 1 and not r2['wrote']
    sam = spark_collect.read.format('delta').load(tables + '/sam')
    assert sam.count() == 2
    x_ids = {
        r.fp_int
        for r in spark_collect.read.format('delta').load(tables + '/X')
        .select('fp_int').distinct().collect()
    }
    assert len(x_ids) == 2


def test_ingest_run_force_rewrites_one(dummy_h5ad_pair, spark_collect, tmp_path):
    paths = _link_pair(dummy_h5ad_pair, tmp_path / 'h5ads')
    tables = str(tmp_path / 'tables')
    kw = _run_kw(spark_collect, str(tmp_path / 'h5ads'), tables)
    ingest.run(**kw)
    before = {
        r.file_path: r.ingested_at
        for r in spark_collect.read.format('delta').load(tables + '/sam').collect()
    }
    r2 = ingest.run(**kw, force=paths[0])
    assert r2['n_forced'] == 1 and r2['n_skipped'] == 1 and r2['wrote']
    after = {
        r.file_path: r.ingested_at
        for r in spark_collect.read.format('delta').load(tables + '/sam').collect()
    }
    assert after[paths[0]] != before[paths[0]]
    assert after[paths[1]] == before[paths[1]]
