import pytest

from cspray.data import SprayData
from cspray.write import fp_int_predicate
from pyspark.sql import functions as F

def test_write_order_sam_last():
    """X first (fail-fast); sam last so a crash still looks uningested."""
    sdata = SprayData()
    assert sdata.write_order[0] == 'X'
    assert sdata.write_order[-1] == 'sam'
    assert sdata.write_order == ['X', 'var', 'obs', 'clu', 'sta', 'sam']


def _spy_writes():
    written = []
    def write_fn(spark, df, name, partition=False, cluster=False, replace_where=None):
        written.append({
            'name': name,
            'partition': partition,
            'cluster': cluster,
            'replace_where': replace_where,
            'columns': list(df.columns),
        })
    return written, write_fn


def test_to_tables_writes_present_frames_in_order(dummy_h5ad_pair, spark_collect):
    """None slots (clu/sta) are skipped; X first, sam last."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    written, spy = _spy_writes()
    out = sdata.to_tables(
        spark_collect, write_fn=spy, table_base='t', join_char='_'
    )
    assert out == ['X', 'var', 'obs', 'sam']
    assert [w['name'] for w in written] == ['t_X', 't_var', 't_obs', 't_sam']
    assert all(not w['partition'] and not w['cluster'] for w in written)


def test_to_tables_writes_extra_slot_before_sam(dummy_h5ad_pair, spark_collect):
    """A later table (obsm/layers) just joins write_order ahead of sam."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    sdata.obsm = sdata.obs
    sdata.write_order = ['X', 'var', 'obs', 'obsm', 'clu', 'sta', 'sam']
    written, spy = _spy_writes()
    out = sdata.to_tables(
        spark_collect, write_fn=spy, table_base='t', join_char='_'
    )
    assert out == ['X', 'var', 'obs', 'obsm', 'sam']
    assert out[0] == 'X'
    assert out[-1] == 'sam'
    assert out.index('obsm') < out.index('sam')
    assert [w['name'] for w in written] == [
        't_X', 't_var', 't_obs', 't_obsm', 't_sam'
    ]


def test_to_tables_partition_cluster_follow_columns(dummy_h5ad_pair, spark_collect):
    """Flags apply by column, not by a hardcoded ['X','obs'] name list."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    written, spy = _spy_writes()
    sdata.to_tables(
        spark_collect, write_fn=spy, table_base='t', join_char='_',
        partition=True, cluster=True,
    )
    by = {w['name']: w for w in written}
    assert by['t_X']['partition'] and by['t_X']['cluster']
    assert by['t_obs']['partition'] and by['t_obs']['cluster']
    assert by['t_var']['partition'] and not by['t_var']['cluster']
    assert by['t_sam']['partition'] and not by['t_sam']['cluster']


def test_to_tables_subset_unknown_raises():
    sdata = SprayData()
    with pytest.raises(ValueError, match='not in write_order'):
        sdata.to_tables(None, write_fn=lambda *a, **k: None, subset=['obsm'])

def test_fp_int_predicate_range_and_in():
    assert fp_int_predicate([7]) == 'fp_int = 7'
    assert fp_int_predicate([10, 11, 12]) == 'fp_int BETWEEN 10 AND 12'
    assert fp_int_predicate([1, 3, 5]) == 'fp_int IN (1, 3, 5)'
    assert fp_int_predicate([0, 1, 2, 5, 9]) == (
        '(fp_int BETWEEN 0 AND 2) OR (fp_int IN (5, 9))'
    )


def test_fp_int_predicate_empty_raises():
    with pytest.raises(ValueError, match='at least one fp_int'):
        fp_int_predicate([])


def test_to_tables_default_overwrite_wipes_other_files(
    dummy_h5ad_pair, spark_collect, tmp_path
):
    """No replace_where still replaces the whole table."""
    spark = spark_collect
    table_base = str(tmp_path)
    sdata = SprayData.from_h5ads(
        spark, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    sdata.to_tables(spark, table_base=table_base, join_char='/')

    keep = 0
    sdata.X = sdata.X.filter(F.col('fp_int') == keep)
    sdata.obs = sdata.obs.filter(F.col('fp_int') == keep)
    sdata.var = sdata.var.filter(F.col('fp_int') == keep)
    sdata.sam = sdata.sam.filter(F.col('fp_int') == keep)
    sdata.to_tables(spark, table_base=table_base, join_char='/')

    loaded = SprayData.from_tables(
        spark, table_base=table_base, join_char='/', mode='delta'
    )
    assert {r.fp_int for r in loaded.X.select('fp_int').distinct().collect()} == {keep}
    assert {r.fp_int for r in loaded.sam.select('fp_int').distinct().collect()} == {keep}


def test_replace_where_keeps_other_fp_int(
    dummy_h5ad_pair, spark_collect, tmp_path
):
    """replaceWhere on one fp_int leaves the other file's rows in place."""
    spark = spark_collect
    table_base = str(tmp_path)
    sdata = SprayData.from_h5ads(
        spark, path=dummy_h5ad_pair, from_raw=False,
        mode='delta', force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    sdata.to_tables(spark, table_base=table_base, join_char='/')

    ids = sorted({r.fp_int for r in sdata.sam.select('fp_int').collect()})
    replace_id, keep_id = ids[0], ids[1]
    keep_obs = sdata.obs.filter(F.col('fp_int') == keep_id).count()

    sdata.X = (
        sdata.X.filter(F.col('fp_int') == replace_id)
        .withColumn('expression', F.lit(99.0).cast('float'))
    )
    sdata.obs = sdata.obs.filter(F.col('fp_int') == replace_id)
    sdata.var = sdata.var.filter(F.col('fp_int') == replace_id)
    sdata.sam = sdata.sam.filter(F.col('fp_int') == replace_id)
    sdata.to_tables_and_reset(
        spark, table_base=table_base, join_char='/',
        replace_where=fp_int_predicate([replace_id]),
    )

    assert {r.fp_int for r in sdata.X.select('fp_int').distinct().collect()} == {
        replace_id, keep_id
    }
    exprs = {
        r.expression
        for r in sdata.X.filter(F.col('fp_int') == replace_id)
        .select('expression').distinct().collect()
    }
    assert exprs == {99.0}
    assert sdata.obs.filter(F.col('fp_int') == keep_id).count() == keep_obs
    keep_exprs = {
        r.expression
        for r in sdata.X.filter(F.col('fp_int') == keep_id)
        .select('expression').distinct().collect()
    }
    assert 99.0 not in keep_exprs
