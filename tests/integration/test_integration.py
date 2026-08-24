import os
import json
import math
import pytest
from scipy.sparse import coo_matrix
from scipy import stats
import numpy as np
import pandas as pd

from cspray.data import SprayData
from cspray.read import _explode_nnz_ranges, construct_h5ad_path_df
import cspray as cs

# --- metadata round-trip comparison helpers --------------------------------
# The obs/var metadata payload is serialized to JSON (VARIANT/string), which is
# intentionally lossy in predictable ways: NaN/NA/NaT -> null, one JSON number
# type, bytes vs str, float precision. Normalize both sides before comparing.
_MISSING = object()

def _norm(v):
    if v is None:
        return _MISSING
    try:
        if pd.isna(v):
            return _MISSING          # np.nan, pd.NA, NaT
    except (TypeError, ValueError):
        pass                         # pd.isna raises on some non-scalars
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode()
    if isinstance(v, (bool, np.bool_)):   # bool before int (bool is a subclass)
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)                   # JSON has a single number type
    return str(v)

def _values_match(a, b):
    na, nb = _norm(a), _norm(b)
    if na is _MISSING or nb is _MISSING:
        return na is nb                   # match only if both missing
    if isinstance(na, float) and isinstance(nb, float):
        return math.isclose(na, nb, rel_tol=1e-9, abs_tol=1e-12)
    return na == nb

# to be placed in SprayData later as part of to_anndata method
def sdata_to_csr(sdata, expression_col = 'expression'):
    # assumes sdata only has one file
    pdf = sdata.X.select('cell_idx','gene_idx',expression_col).toPandas()
    coo = coo_matrix(
        (pdf[expression_col], (pdf['cell_idx'], pdf['gene_idx'])),
        shape = (sdata.obs.count(), sdata.var.count())   
    )
    csr = coo.tocsr()
    return csr

def compare_csr_adata(csr,adata):
    # same number of entries?
    assert csr.nnz == adata.X.nnz
    # confirm the values in adata and csr are same
    assert (adata.X != csr).nnz == 0 

def test_file_exists(downloaded_file):
    assert os.path.exists(downloaded_file)

def test_file_size(downloaded_file):
    assert os.path.getsize(downloaded_file) > 0

def test_cspray_read_success(cspray_read_stage):
    sdata = cspray_read_stage
    try:
        count = sdata.X.count()
        assert count > 0
    except Exception as e:
        print(f"cspray_read_stage failed: {e}")
        assert False

def test_cspray_scanpy_shape_match(cspray_read_stage, scanpy_read_stage):
    sdata = cspray_read_stage
    adata = scanpy_read_stage
    
    assert (sdata.obs.count(), sdata.var.count()) == tuple(adata.X.shape)

def test_cspray_metadata_chunked(downloaded_file, spark_collect, scanpy_read_stage):
    """Read obs/var metadata with a small chunk size (forcing multiple chunks per
    file) and confirm chunking preserves counts, produces contiguous global
    indices, and round-trips the metadata payload.

    Uses metadata_variant=False (JSON string) so it is portable to OSS Spark,
    which lacks the VARIANT type / parse_json.
    """
    adata = scanpy_read_stage
    n_cells, n_genes = adata.X.shape

    sdata = SprayData.from_h5ads(
        spark_collect,
        path=downloaded_file,
        force_partitioning=4,
        from_raw=False,
        mode='delta',
        obs_metadata_columns='all',
        var_metadata_columns='all',
        metadata_variant=False,      # JSON string column: portable to OSS Spark
        metadata_chunk_size=100,     # << n_cells/n_genes so the file splits into many chunks
    )

    # metadata columns are present
    assert 'obs_data' in sdata.obs.columns
    assert 'var_data' in sdata.var.columns

    # chunking preserved every cell with a contiguous, unique global cell_idx
    cell_idx = sdata.obs.select('cell_idx').toPandas()['cell_idx'].values
    assert len(cell_idx) == n_cells
    assert set(cell_idx) == set(range(n_cells))

    # ... and every gene with a contiguous, unique global gene_idx
    gene_idx = sdata.var.select('gene_idx').toPandas()['gene_idx'].values
    assert len(gene_idx) == n_genes
    assert set(gene_idx) == set(range(n_genes))

    # metadata payload round-trips to a JSON object per row
    sample = sdata.obs.select('obs_data').limit(1).toPandas()['obs_data'].iloc[0]
    assert isinstance(json.loads(sample), dict)


def test_cspray_metadata_profile_and_promote(downloaded_file, spark_collect):
    """cs.md.profile summarizes the obs/var metadata payload (key coverage,
    cardinality, suggested actions) and cs.md.promote materializes chosen keys.

    Uses metadata_variant=False (JSON string) so the whole path is portable to
    OSS Spark: profile normalizes to JSON text and profiles via mapInPandas +
    standard aggregations, and promote uses get_json_object.
    """
    sdata = SprayData.from_h5ads(
        spark_collect,
        path=downloaded_file,
        force_partitioning=4,
        from_raw=False,
        mode='delta',
        obs_metadata_columns='all',
        var_metadata_columns='all',
        metadata_variant=False,
    )

    n_cells = sdata.obs.count()

    # --- profile obs -------------------------------------------------------
    report = cs.md.profile(sdata, which='obs', verbose=False)
    assert isinstance(report, pd.DataFrame)
    assert not report.empty
    expected_cols = {
        'key', 'present', 'missing', 'coverage_pct', 'tag',
        'n_distinct', 'value_types', 'type_conflict',
        'possible_alias', 'suggested_action',
    }
    assert expected_cols.issubset(set(report.columns))

    # coverage arithmetic is internally consistent
    assert (report['present'] + report['missing'] == n_cells).all()
    assert report['coverage_pct'].max() <= 100.0
    # keys actually present in the raw obs payload show up in the report
    obs_keys = set(
        json.loads(sdata.obs.select('obs_data').limit(1).toPandas()['obs_data'].iloc[0]).keys()
    )
    assert obs_keys.issubset(set(report['key']))

    # --- per_file view returns a (summary, matrix) tuple -------------------
    summary, matrix = cs.md.profile(sdata, which='var', per_file=True, verbose=False)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(matrix, pd.DataFrame)

    # --- promote a key into a top-level column -----------------------------
    key = report['key'].iloc[0]
    cs.md.promote(sdata, key, which='obs')
    assert key in sdata.obs.columns
    # one value per cell, extracted as a (string) column
    assert sdata.obs.select(key).count() == n_cells

def test_metadata_dummy_roundtrip(dummy_h5ad_pair, spark_collect):
    """End-to-end metadata check on two hand-built h5ad files with known values:
    pick-up, exact per-row assignment, coverage/action classification, per-file
    localisation, and promotion. Uses metadata_variant=False for OSS portability.

    max_categorical=3 makes the id-like/categorical split deterministic at this
    tiny scale (the default present>50 id-like rule cannot fire on 6 rows).
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )

    # 1. sam exists immediately at one row per input file -------------------
    sam = sdata.sam.toPandas()
    assert len(sam) == 2
    assert set(sam.columns) == {'fp_int', 'file_path'}
    assert set(sam['file_path']) == set(dummy_h5ad_pair)

    # 2. every cell carries metadata, keyed to the right barcode ------------
    obs = sdata.obs.select('cell_barcode', 'obs_data').toPandas()
    assert set(obs['cell_barcode']) == {'cA0','cA1','cA2','cB0','cB1','cB2'}
    parsed = {r.cell_barcode: json.loads(r.obs_data) for r in obs.itertuples()}
    assert parsed['cA0']['cell_type'] == 'T'
    assert parsed['cA0']['n_genes'] == 10
    assert parsed['cB2']['cell_type'] == 'B'
    assert parsed['cB2']['batch'] == 'b2'
    assert 'batch' not in parsed['cA0']       # file A never had it
    assert 'CellType' not in parsed['cB0']    # file B never had it

    # 3. profile: pick-up + coverage + classification -----------------------
    r = cs.md.profile(sdata, which='obs', max_categorical=3, verbose=False).set_index('key')
    assert set(r.index) == {
        'cell_type','tissue','donor_id','organism','n_genes','qc_score',
        'CellType','batch','mixed'}
    assert r.loc['cell_type','coverage_pct'] == 100.0 and r.loc['cell_type','present'] == 6
    assert r.loc['batch','coverage_pct'] == 50.0 and r.loc['batch','tag'] == 'partial'
    assert r.loc['CellType','coverage_pct'] == 50.0

    assert r.loc['organism','suggested_action'] == 'drop_constant'
    assert r.loc['cell_type','suggested_action'] == 'promote'
    assert r.loc['donor_id','suggested_action'] == 'keep_in_variant'   # nd=6 > 3
    assert bool(r.loc['cell_type','possible_alias']) and bool(r.loc['CellType','possible_alias'])
    assert bool(r.loc['mixed','type_conflict'])

    # 4. per-file matrix -----------------------------------------------------
    _, matrix = cs.md.profile(sdata, which='obs', per_file=True, verbose=False)
    assert matrix.shape[1] == 2
    assert set(matrix.loc['batch']) == {0.0, 100.0}
    assert (matrix.loc['cell_type'] == 100.0).all()

    # 5. promote (string + typed) -------------------------------------------
    cs.md.promote(sdata, 'cell_type', which='obs')
    cs.md.promote(sdata, 'n_genes', which='obs', dtypes='int')
    prom = sdata.obs.select('cell_barcode','cell_type','n_genes').toPandas().set_index('cell_barcode')
    assert prom.loc['cA0','cell_type'] == 'T'
    assert int(prom.loc['cA1','n_genes']) == 20
    assert int(prom.loc['cB2','n_genes']) == 35

    # 6. var side, briefly ---------------------------------------------------
    vr = cs.md.profile(sdata, which='var', max_categorical=3, verbose=False).set_index('key')
    assert {'feature_type','highly_variable'}.issubset(set(vr.index))
    assert vr.loc['feature_type','suggested_action'] == 'drop_constant'


def test_sample_metadata_promotion_and_qc_order(dummy_h5ad_pair, spark_collect):
    """Sample metadata and QC independently enrich the read-time sam table."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)

    plan = cs.md.promote_sample_suggested(sdata, dry_run=True, verbose=False)
    assert set(plan.columns) == {'which', 'key', 'dtype', 'suggested_action'}
    assert set(plan['which']) == {'sample'}
    assert {'tissue', 'organism'}.issubset(set(plan['key']))
    assert 'cell_type' not in set(plan['key'])
    assert 'donor_id' not in set(plan['key'])

    # If a sample-level key was previously promoted on obs, moving it to sam
    # removes only that duplicated top-level column (the payload remains).
    cs.md.promote(sdata, 'tissue', which='obs')
    assert 'tissue' in sdata.obs.columns
    cs.md.promote_sample_suggested(sdata, verbose=False)
    sam = sdata.sam.select('file_path', 'tissue', 'organism').toPandas().set_index('file_path')
    assert sam.loc[dummy_h5ad_pair[0], 'tissue'] == 'lung'
    assert sam.loc[dummy_h5ad_pair[1], 'tissue'] == 'blood'
    assert set(sam['organism']) == {'human'}
    assert 'tissue' not in sdata.obs.columns
    assert 'tissue' in json.loads(
        sdata.obs.select('obs_data').limit(1).toPandas()['obs_data'].iloc[0]
    )

    cs.pp.calculate_qc_metrics(sdata)
    assert {'tissue', 'organism', 'n_cells', 'total_counts',
            'mean_genes_per_cell'}.issubset(set(sdata.sam.columns))

    external = spark_collect.createDataFrame([
        (dummy_h5ad_pair[0], 'study-a'),
        (dummy_h5ad_pair[1], 'study-b'),
    ], ['file_path', 'study'])
    cs.md.add_sample_metadata(sdata, external)
    assert set(sdata.sam.select('study').toPandas()['study']) == {'study-a', 'study-b'}


def test_qc_before_sample_metadata_promotion(dummy_h5ad_pair, spark_collect):
    """Promoting after QC preserves previously calculated sample metrics."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    cs.pp.calculate_qc_metrics(sdata)
    cs.md.promote_sample(sdata, ['tissue'], dtypes='string')

    assert {'tissue', 'n_cells', 'total_counts',
            'mean_genes_per_cell'}.issubset(set(sdata.sam.columns))
    assert set(sdata.sam.select('tissue').toPandas()['tissue']) == {'lung', 'blood'}


def test_promote_suggested_with_promote_sam(dummy_h5ad_pair, spark_collect):
    """promote_sam=True routes sample-constant keys to sam in the same call, and
    keeps them out of the obs plan so nothing is materialized per cell."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)

    plan = cs.md.promote_suggested(
        sdata, which='both', profile_kwargs={'max_categorical': 3},
        promote_sam=True, dry_run=True, verbose=False,
    )
    by_target = {w: set(g['key']) for w, g in plan.groupby('which')}
    assert {'tissue', 'organism'}.issubset(by_target['sample'])
    assert 'tissue' not in by_target['obs']       # excluded in favour of sam
    assert 'cell_type' in by_target['obs']        # varies per cell, stays on obs
    assert 'highly_variable' in by_target['var']

    cs.md.promote_suggested(
        sdata, which='both', profile_kwargs={'max_categorical': 3},
        promote_sam=True, verbose=False,
    )
    assert 'cell_type' in sdata.obs.columns
    assert 'tissue' not in sdata.obs.columns
    sam = sdata.sam.select('file_path', 'tissue').toPandas().set_index('file_path')
    assert sam.loc[dummy_h5ad_pair[0], 'tissue'] == 'lung'
    assert sam.loc[dummy_h5ad_pair[1], 'tissue'] == 'blood'

    with pytest.raises(ValueError):
        cs.md.promote_suggested(sdata, which='obs', keys=['tissue'], promote_sam=True)


def test_metadata_promote_suggested(dummy_h5ad_pair, spark_collect):
    """promote_suggested: dry_run returns an editable plan without mutating, and
    applying it materializes the recommended keys with inferred Spark types.

    max_categorical=3 (via profile_kwargs) makes the classification deterministic
    at this tiny scale, so cell_type and tissue (obs) and highly_variable (var)
    are 'promote'; organism is drop_constant and the unique/mixed keys
    keep_in_variant. tissue is sample-constant, so it is only an obs candidate
    while promote_sam is off (see test_promote_suggested_with_promote_sam).
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )

    # --- dry_run: plan only, no mutation -----------------------------------
    plan = cs.md.promote_suggested(
        sdata, which='obs', profile_kwargs={'max_categorical': 3},
        dry_run=True, verbose=False,
    )
    assert isinstance(plan, pd.DataFrame)
    assert set(plan.columns) == {'which', 'key', 'dtype', 'suggested_action'}
    plan_keys = set(plan['key'])
    assert 'cell_type' in plan_keys        # promote
    assert 'tissue' in plan_keys           # promote (sample-constant, but promote_sam off)
    assert 'organism' not in plan_keys     # drop_constant
    assert 'donor_id' not in plan_keys     # keep_in_variant (nd=6 > 3)
    assert plan.set_index('key').loc['cell_type', 'dtype'] == 'string'
    assert 'cell_type' not in sdata.obs.columns   # dry_run must not mutate

    # --- apply across both axes, typed -------------------------------------
    cs.md.promote_suggested(
        sdata, which='both', profile_kwargs={'max_categorical': 3}, verbose=False,
    )
    assert 'cell_type' in sdata.obs.columns          # obs promote
    assert 'highly_variable' in sdata.var.columns    # var promote
    # inferred types: str -> string, bool -> boolean
    assert sdata.obs.schema['cell_type'].dataType.simpleString() == 'string'
    assert sdata.var.schema['highly_variable'].dataType.simpleString() == 'boolean'
    obs = sdata.obs.select('cell_barcode', 'cell_type').toPandas().set_index('cell_barcode')
    assert obs.loc['cA0', 'cell_type'] == 'T'

    # --- guard: keys override not allowed with which='both' ----------------
    with pytest.raises(ValueError):
        cs.md.promote_suggested(sdata, which='both', keys=['cell_type'])

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

@pytest.mark.parametrize("metadata_chunk_size", [100,])
def test_metadata_matches_scanpy_chunked(downloaded_file, spark_collect, scanpy_read_stage, metadata_chunk_size):
    """Strongest metadata oracle: read the full file with a chunk size far smaller
    than the number of rows (forcing many chunk boundaries) and confirm every
    obs/var column value matches scanpy's AnnData for every row, keyed by
    barcode / gene id. Guards value-level correctness of chunked ingestion
    (incl. the bounded-categorical decode), not just counts/indices.
    """
    adata = scanpy_read_stage
    sdata = SprayData.from_h5ads(
        spark_collect,
        path=downloaded_file,
        from_raw=False,
        mode='delta',
        obs_metadata_columns='all',
        var_metadata_columns='all',
        metadata_variant=False,
        metadata_chunk_size=metadata_chunk_size,
    )

    # --- obs: compare every column of every cell, keyed by barcode ----------
    obs_got = sdata.obs.select('cell_barcode', 'obs_data').toPandas()
    assert len(obs_got) == adata.n_obs
    assert set(obs_got['cell_barcode']) == set(map(str, adata.obs.index))
    # payload keys cover exactly the (non-index) obs columns
    sample_keys = set(json.loads(obs_got['obs_data'].iloc[0]).keys())
    assert sample_keys == set(adata.obs.columns)

    for bc, payload in ((r.cell_barcode, json.loads(r.obs_data)) for r in obs_got.itertuples()):
        expected_row = adata.obs.loc[bc]
        for col, got_val in payload.items():
            assert _values_match(got_val, expected_row[col]), (
                f"obs mismatch bc={bc} col={col}: got={got_val!r} exp={expected_row[col]!r}"
            )

    # --- var: same check, keyed by gene id ---------------------------------
    var_got = sdata.var.select('gene_id', 'var_data').toPandas()
    assert set(var_got['gene_id']) == set(map(str, adata.var.index))
    for gid, payload in ((r.gene_id, json.loads(r.var_data)) for r in var_got.itertuples()):
        expected_row = adata.var.loc[gid]
        for col, got_val in payload.items():
            if col not in adata.var.columns:
                continue   # cspray may carry a derived gene_name col not in raw var
            assert _values_match(got_val, expected_row[col]), (
                f"var mismatch gid={gid} col={col}: got={got_val!r} exp={expected_row[col]!r}"
            )

def test_cspray_scanpy_expression_match(cspray_read_stage, scanpy_read_stage):
    sdata = cspray_read_stage
    adata = scanpy_read_stage
    csr = sdata_to_csr(sdata)
    compare_csr_adata(csr,adata)

def test_cspray_scanpy_pp_match(cspray_pp_stage, scanpy_pp_stage):
    sdata = cspray_pp_stage
    adata = scanpy_pp_stage

    print('adata obs cols: ', adata.obs.columns)
    print('adata var cols: ', adata.var.columns)

    # did I get same cells and same genes as pp os only filtering not changing any values
    scanpy_set = set(adata.obs['int_idx'].values)
    cspray_set = set(sdata.obs.select(['cell_idx']).toPandas()['cell_idx'].values)
    assert scanpy_set == cspray_set

    # gene filtering confirm
    print("these are the var columns.....")
    print(adata.var.columns)
    scanpy_g_set = set(adata.var['int_idx'].values)
    cspray_g_set = set(sdata.var.select(['gene_idx']).toPandas()['gene_idx'].values)
    print("scanpy genes : ",len(scanpy_g_set))
    print("cspray genes : ",len(cspray_g_set))
    assert scanpy_g_set == cspray_g_set



# HVG test 
def test_hvg(spark_collect, cspray_hvg_stage, scanpy_hvg_stage):
    sdata = cspray_hvg_stage
    adata = scanpy_hvg_stage
    spark = spark_collect

    sdf = sdata.sta.filter(sdata.sta.selected==True)\
    .orderBy('z_dispersion', ascending=False)\
    .select('gene_name','log1p_mean')\
    .withColumnsRenamed({
        'gene_name':'Gene name',
    }).join(
        spark.createDataFrame(adata.var[adata.var.highly_variable==True].sort_values('dispersions_norm',ascending=False)[['Gene name','means']]),
        on='Gene name'
    )

    pdf = sdf.toPandas()
    allclose = np.allclose(
        pdf['means'],
        pdf['log1p_mean'],
        equal_nan=True,
        rtol=2e-2
    )
    print(np.mean(np.isclose(
        pdf['means'],
        pdf['log1p_mean'],
        equal_nan=True,
        rtol=2e-2)
    ))
    print(f"allclose (means) = {allclose}")
    assert allclose

    pr = stats.pearsonr(sdf.select('log1p_mean').toPandas().values.flatten(),sdf.select('means').toPandas().values.flatten()).statistic
    print(f"pearsons = {pr}")
    assert pr > 0.98
    
    
    sdf = sdata.sta.filter(sdata.sta.selected==True)\
    .orderBy('z_dispersion', ascending=False)\
    .select('gene_name','z_dispersion')\
    .withColumnsRenamed({
        'gene_name':'Gene name',
    }).join(
        spark.createDataFrame(adata.var[adata.var.highly_variable==True].sort_values('dispersions_norm',ascending=False)[['Gene name','dispersions_norm']]),
        on='Gene name'
    )
    overlap = sdf.count()
    print(f"overlap = {overlap}")
    # is there at least 450/500 matching genes in the HVGs selected
    assert sdf.count() > 450

    pdf = sdf.toPandas()
    isclose = np.isclose(
        pdf['dispersions_norm'],
        pdf['z_dispersion'],
        equal_nan=True, # some can be missing, we accept minor variation
        rtol=2e-2,
    )
    print(np.mean(isclose))
    assert np.mean(isclose)>0.92 #92% of the genes are within tolerance (some smaller dispersion cases in small sample size may differ by more, ok with that if ensure pearson)
    
    pr = stats.pearsonr(sdf.select('z_dispersion').toPandas().values.flatten(),sdf.select('dispersions_norm').toPandas().values.flatten()).statistic
    print(f"pearsons = {pr}")
    assert pr > 0.98

def test_cspray_standard_runthrough(spark_collect, cspray_final_stage):
    sdata = cspray_final_stage
    spark = spark_collect

    sam = sdata.sam.toPandas()
    clu = sdata.clu.toPandas()

    print(sam)
    print(clu)
    
    assert len(clu) >= 2 # expected number of found clusters
    assert len(clu) <= 3 # expected number of found clusters
    
    assert len(sam) == 1 # number of samples

    assert np.round(sam.iloc[0]['n_cells']) == 500
    # file used is changed
    # assert np.round(sam.iloc[0]['mean_genes_per_cell']) == 1587
    # assert np.isclose( sam.iloc[0]['pct_cells_passing_mt_8.0_pct'], 0.5080321285140562)









    