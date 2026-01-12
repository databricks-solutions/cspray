import os
from scipy.sparse import coo_matrix
from scipy import stats
import numpy as np

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

def test_cspray_scanpy_expression_match(cspray_read_stage, scanpy_read_stage):
    sdata = cspray_read_stage
    adata = scanpy_read_stage
    csr = sdata_to_csr(sdata)
    compare_csr_adata(csr,adata)

def test_cspray_scanpy_pp_match(cspray_pp_stage, scanpy_pp_stage):
    sdata = cspray_pp_stage
    adata = scanpy_pp_stage

    # did I get same cells and same genes as pp os only filtering not changing any values
    scanpy_set = set(adata.obs['int_idx'].values)
    cspray_set = set(sdata.obs.select(['cell_idx']).toPandas()['cell_idx'].values)
    assert scanpy_set == cspray_set

    # gene filtering confirm
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
    assert np.mean(isclose)>0.95 #95% of the genes are within tolerance (some smaller dispersion cases in small sample size may differ by more, ok with that if ensure pearson)
    
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
    assert np.round(sam.iloc[0]['mean_genes_per_cell']) == 1587
    assert np.isclose( sam.iloc[0]['pct_cells_passing_mt_8.0_pct'], 0.5080321285140562)









    