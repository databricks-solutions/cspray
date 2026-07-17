import os
import json
from scipy.sparse import coo_matrix
from scipy import stats
import numpy as np
import pandas as pd

from cspray.data import SprayData
import cspray as cs

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

    # 1. every cell carries metadata, keyed to the right barcode ------------
    obs = sdata.obs.select('cell_barcode', 'obs_data').toPandas()
    assert set(obs['cell_barcode']) == {'cA0','cA1','cA2','cB0','cB1','cB2'}
    parsed = {r.cell_barcode: json.loads(r.obs_data) for r in obs.itertuples()}
    assert parsed['cA0']['cell_type'] == 'T'
    assert parsed['cA0']['n_genes'] == 10
    assert parsed['cB2']['cell_type'] == 'B'
    assert parsed['cB2']['batch'] == 'b2'
    assert 'batch' not in parsed['cA0']       # file A never had it
    assert 'CellType' not in parsed['cB0']    # file B never had it

    # 2. profile: pick-up + coverage + classification -----------------------
    r = cs.md.profile(sdata, which='obs', max_categorical=3, verbose=False).set_index('key')
    assert set(r.index) == {
        'cell_type','donor_id','organism','n_genes','qc_score','CellType','batch','mixed'}
    assert r.loc['cell_type','coverage_pct'] == 100.0 and r.loc['cell_type','present'] == 6
    assert r.loc['batch','coverage_pct'] == 50.0 and r.loc['batch','tag'] == 'partial'
    assert r.loc['CellType','coverage_pct'] == 50.0

    assert r.loc['organism','suggested_action'] == 'drop_constant'
    assert r.loc['cell_type','suggested_action'] == 'promote'
    assert r.loc['donor_id','suggested_action'] == 'keep_in_variant'   # nd=6 > 3
    assert bool(r.loc['cell_type','possible_alias']) and bool(r.loc['CellType','possible_alias'])
    assert bool(r.loc['mixed','type_conflict'])

    # 3. per-file matrix -----------------------------------------------------
    _, matrix = cs.md.profile(sdata, which='obs', per_file=True, verbose=False)
    assert matrix.shape[1] == 2
    assert set(matrix.loc['batch']) == {0.0, 100.0}
    assert (matrix.loc['cell_type'] == 100.0).all()

    # 4. promote (string + typed) -------------------------------------------
    cs.md.promote(sdata, 'cell_type', which='obs')
    cs.md.promote(sdata, 'n_genes', which='obs', dtypes='int')
    prom = sdata.obs.select('cell_barcode','cell_type','n_genes').toPandas().set_index('cell_barcode')
    assert prom.loc['cA0','cell_type'] == 'T'
    assert int(prom.loc['cA1','n_genes']) == 20
    assert int(prom.loc['cB2','n_genes']) == 35

    # 5. var side, briefly ---------------------------------------------------
    vr = cs.md.profile(sdata, which='var', max_categorical=3, verbose=False).set_index('key')
    assert {'feature_type','highly_variable'}.issubset(set(vr.index))
    assert vr.loc['feature_type','suggested_action'] == 'drop_constant'

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









    