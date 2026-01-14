import pytest
import requests
import tempfile
import os
import shutil

import gget
import cellxgene_census

import pooch
import scanpy as sc
from cspray.data import SprayData
import cspray as cs
import pandas as pd
import numpy as np

# def my_write_fn(df, name:str):
#     df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(name)

# def my_read_fn(spark, name:str):
#     return spark.read.format("delta").load(name)

@pytest.fixture(scope="module")
def downloaded_file():

    path = "/tmp/filtered_feature_bc_matrix.h5ad"
    
    if os.path.exists(path):
        os.remove(path)

    gget.setup("cellxgene")
    
    cellxgene_census.download_source_h5ad(
        dataset_id = '0de831e0-a525-4ec5-b717-df56f2de2bf0', 
        to_path = path, 
        census_version = "2025-01-30",
        progress_bar=False,
    )
    sample_adata = sc.read_h5ad(path)
    
    # here we use adata.raw if it exists as is assumeed to be unprocessed raw data
    if "_raw" in sample_adata.__dict__:
        if sample_adata._raw is not None:
            sample_adata = sample_adata.raw.to_adata()

    # take small set only for speed of testing
    sample_adata = sample_adata[:500,:]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    sample_adata.write(temp_file.name)
    
    yield temp_file.name
        
    os.remove(temp_file.name)

@pytest.fixture(scope="module")
def scanpy_read_stage(downloaded_file):
    adata = sc.read_h5ad(
        downloaded_file,
    )
    return adata

@pytest.fixture(scope="module")
def scanpy_pp_stage(scanpy_read_stage):

    adata = scanpy_read_stage

    gene_col = 'index'

    # for later testing
    adata.obs['int_idx'] = np.arange(len(adata.obs))
    adata.var['int_idx'] = np.arange(len(adata.var))

    adata.var = adata.var.reset_index()
    # adata.var['index'] = adata.var['index'].astype(str) 
    adata.var['Gene name'] = adata.var['index']
    adata.var = adata.var.set_index('index')

    adata.var["mt"] = adata.var['Gene name'].str.startswith("MT-",na=False)
    # ribosomal genes
    adata.var["ribo"] = adata.var['Gene name'].str.startswith(("RPS", "RPL"),na=False)
    # hemoglobin genes
    adata.var["hb"] = adata.var['Gene name'].str.contains("^HB[^(P)]",na=False)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )

    print("start filter")
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.pct_counts_mt < 8, :]
    return adata

@pytest.fixture(scope="module")
def scanpy_ppnorm_stage(scanpy_pp_stage):

    adata = scanpy_pp_stage

    sc.pp.normalize_total(adata,target_sum=1e4)
    return adata

@pytest.fixture(scope="module")
def scanpy_ppnormlog_stage(scanpy_ppnorm_stage):

    adata = scanpy_ppnorm_stage

    sc.pp.log1p(adata)
    return adata

@pytest.fixture(scope="module")
def scanpy_hvg_stage(scanpy_ppnormlog_stage):

    adata = scanpy_ppnormlog_stage

    sc.pp.highly_variable_genes(adata, n_top_genes=500)
    return adata

####################################################################
################ CSPRAY FIXTURES ################################
####################################################################

@pytest.fixture(scope="module")
def spark_collect():
    """ if no existing spark session, start one

    you may wish to customize the config if using your own spark session
    - ensure you additionally install spark dependencies on top the pip requirements etc
        - i.e get Java, and set env vars JAVA_HOME and SPARK_HOME
    """
    if 'spark' not in globals():
        print('creating new spark session (importing sparksession, make spark session, set global spark var)')
        from pyspark.sql import SparkSession
        # global spark
        # set driver memory higher for single driver local execution (assumes you have at least 12gb RAM)
        spark = (SparkSession.builder
            .appName("CsprayWorkflow")
            .master("local[*]")
            .config("spark.driver.memory", "6g")
            .getOrCreate()
        )
    else:
        spark=spark
    return spark

@pytest.fixture(scope="module")
def build_dir():
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir.name
    #teardown
    shutil.rmtree(temp_dir.name)

@pytest.fixture(scope="module")
def cspray_read_stage(downloaded_file, spark_collect, build_dir):
    
    sdata = SprayData.from_h5ads(
      spark_collect,
      path=downloaded_file,
      force_partitioning=4, # optional forcing of partitions - only useful for small numbers of files/chunks, set to total cores
      from_raw=False, # we could force reading from raw field in h5ad (normally we would do this for processing the raw counts, here we're just testing an example file without a raw field)
      mode='delta' # use delta tables on disk for storage
    )
    sdata.to_tables_and_reset(
      spark_collect,
      table_base=build_dir, 
      join_char='/bronze_', 
    )
    return sdata

@pytest.fixture(scope="module")
def cspray_pp_stage(cspray_read_stage, spark_collect, build_dir):

    sdata = cspray_read_stage

    cs.pp.calculate_qc_metrics(sdata)
    cs.pp.filter_cells(sdata)
    cs.pp.filter_genes(sdata)
    cs.pp.apply_samplewise_mt_statistic(sdata)
    cs.pp.filter_cells_on_mt(sdata)

    cs.pp.normalize(sdata)
    cs.pp.log1p_counts(sdata)

    sdata.to_tables_and_reset(
        spark_collect,
        table_base=build_dir, 
        join_char='/silver_pp_', 
    )
    return sdata

@pytest.fixture(scope="module")
def cspray_hvg_stage(cspray_pp_stage, spark_collect, build_dir):

    sdata = cspray_pp_stage

    cs.pp.calculate_hvg(sdata, n_hvg=500)

    sdata.to_tables_and_reset(
        spark_collect,
        table_base=build_dir, 
        join_char='/silver_hvg_', 
        subset=['X','var','obs','sta']
    )
    return sdata

@pytest.fixture(scope="module")
def cspray_final_stage(cspray_hvg_stage, spark_collect, build_dir):
    sdata = cspray_hvg_stage
    
    cs.pp.pca(sdata, n_hvg=500)
    sdata.to_tables_and_reset(
        spark_collect,
        table_base=build_dir, 
        join_char='/silver_pca_',
        subset=['obs'] 
    )

    scores_pdf = cs.tl.kmeans(sdata, ks=[2,3])
    sdf_rank = cs.tl.rank_marker_genes(sdata, fc_cutoff=0.15)

    sdata.to_tables_and_reset(
        spark_collect,
        table_base=build_dir, 
        join_char='/gold_',
        subset = ['clu','obs','sam','sta'] # only save key tables
    )
    return sdata










