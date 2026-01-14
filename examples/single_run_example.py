""" This is an example script for executing a cspray workflow on h5ad file(s)

This functions as a simple example, with default paramaters set
 - most functions in preprocessing have additional paramters that can be set

require enviroment with cspray pip installed
----------

usage: single_run_example.py [-h] -r READ_PATH -w WRITE_PATH [--save_bronze_silver] [--predownloaded]

Cspray Workflow

options:
  -h, --help            show this help message and exit
  -r READ_PATH          Path to input h5ad file(s), path to directory if more than one (path to save downloaded h5ad to if not setting --predownloaded)
  -w WRITE_PATH         Path to output Delta table(s)
  --save_bronze_silver  Save bronze and silver tables
  --predownloaded       if set, then assumes data aleady exists, otherwise downloads an h5ad to read_path
  -- scanpy_pca_clu     use naive parallel scanpy operation instead of spark PCA and clustering step (requires a seperate igrpah install)

e.g.:
python single_run_example.py -r /tmp/test.h5ad -w /tmp/test
-----------


"""
from cspray.data import SprayData
import cspray as cs
import pandas as pd
import argparse
import warnings

def parse_args():
  parser = argparse.ArgumentParser(description="Cspray Workflow")
  parser.add_argument('-r', type=str, dest='read_path',  required=True, help='Path to input h5ad file(s), path to directory if more than one (path to save downloaded h5ad to if not setting --predownloaded)')
  parser.add_argument('-w', type=str, dest='write_path', required=True, help='Path to output Delta table(s)')
  parser.add_argument('--scanpy_pca_clu', action='store_true', help='Use naive parallel scanpy operation instead of spark PCA and clustering step')
  parser.add_argument('--no_save_bronze_silver', action='store_true', help='Do not Save bronze and silver tables')
  parser.add_argument('--predownloaded', action='store_true', help='if set, then assumes data aleady exists, otherwise downloads an h5ad to read_path')
  return parser.parse_args()

def download_example(file_path):
  print("------------ downloading sample h5ad file -----------")
  import pooch
  import scanpy as sc
  import requests

  path = "/tmp/filtered_feature_bc_matrix.h5ad"
  import os
  if os.path.exists(path):
    os.remove(path)

  import gget
  import cellxgene_census
  gget.setup("cellxgene")
  
  cellxgene_census.download_source_h5ad(
      dataset_id = '0de831e0-a525-4ec5-b717-df56f2de2bf0', 
      to_path = path, 
      census_version = "2025-01-30",
      progress_bar=False,
  )

  sample_adata = sc.read_h5ad(path)
  sample_adata.var_names_make_unique()

  sample_adata = sample_adata[:1000,:]

  sample_adata.write(file_path)
  return None

def start_spark():
  """ if no existing spark session, start one

  you may wish to customize the config if using your own spark session
    - ensure you additionally install spark dependencies on top the pip requirements etc
      - i.e get Java, and set env vars JAVA_HOME and SPARK_HOME
  """
  if 'spark' not in globals():
    print('creating new spark session (importing sparksession, make spark session, set global spark var)')
    from pyspark.sql import SparkSession
    global spark
    # set driver memory higher for single driver local execution (assumes you have at least 12gb RAM)
    spark = (SparkSession.builder
      .appName("CsprayWorkflow")
      .master("local[*]")
      .config("spark.driver.memory", "6g")
      .getOrCreate()
    )
  return None

def main(
  read_path,
  write_path,
  no_save_bronze_silver=False,
  scanpy_pca_clu=False,
  ):


  # optionally map ensemble_ids to gene name if you don't know the gene_name column and ensembl_ids are in the default index of var
  # reference_gene_table = spark.createDataFrame(cs.utils.get_gene_table())

  
  # read in one or more h5ad files
  sdata = SprayData.from_h5ads(
      spark,
      path=read_path,
      gene_name_column=None, # will make the default index column in 'var' be the gene names, usually set to a known gene name column, or alternatively provde a reference_df to map ensemble_ids to gene_names
      force_partitioning=8, # optional forcing of partitions - only useful for small numbers of files/chunks, set to total cores
      from_raw=False, # we could force reading from raw field in h5ad (normally we would do this for processing the raw counts, here we're just testing an example file without a raw field)
      mode='delta', # writes to delta on disk (not to unity catalog on Databricks)
      # ensembl_reference_df = reference_gene_table # generally preferred
    )
  
  # write output and reset dataframes to be read from tables
  if not no_save_bronze_silver:
    sdata.to_tables_and_reset(
      spark,
      table_base=write_path, 
      join_char='/bronze_', 
    )
  # preprocess
  # additional arguments for parameters are available for most functions
  cs.pp.calculate_qc_metrics(sdata)
  cs.pp.filter_cells(sdata)
  cs.pp.filter_genes(sdata)
  cs.pp.apply_samplewise_mt_statistic(sdata)
  cs.pp.filter_cells_on_mt(sdata)

  cs.pp.normalize(sdata)
  cs.pp.log1p_counts(sdata)


  # write output and reset dataframes to be read from tables
  if not no_save_bronze_silver:
    sdata.to_tables_and_reset(
      spark,
      table_base=write_path, 
      join_char='/silver_pp_', 
    )

  cs.pp.calculate_hvg(sdata, n_hvg=500)

  # write output and reset dataframes to be read from tables
  if not no_save_bronze_silver:
    sdata.to_tables_and_reset(
      spark,
      table_base=write_path, 
      join_char='/silver_hvg_', 
    )

  print("X has N cells : ", sdata.X.select("cell_idx").distinct().count())
  print("obs has       : ", sdata.obs.count())
  print("X has N genes : ", sdata.X.select("gene_idx").distinct().count())
  print("var has       : ", sdata.var.count())

  # perform pca and clustering
  if not scanpy_pca_clu:
    cs.pp.pca(sdata, n_hvg=500)

    if not no_save_bronze_silver:
      sdata.to_tables_and_reset(
        spark,
        table_base=write_path, 
        join_char='/silver_pca_',
        subset=['obs'] 
      )

    scores_pdf = cs.tl.kmeans(sdata, ks=[2,3])
    sdf_rank = cs.tl.rank_marker_genes(sdata, fc_cutoff=0.15)

  else:
    cs.tl.apply_pca_cluster_markers(sdata,cluster_resolution=0.05,cache_intermediary=False)
  
  
  sdata.to_tables_and_reset(
    spark,
    table_base=write_path, 
    join_char='/silver_end_',
    subset = ['clu','obs','sam','sta'] # only save key tables
  )
  
  cs.tl.as_gold_mart_data(sdata)

  sdata.to_tables_and_reset(
    spark,
    table_base=write_path, 
    join_char='/gold_'
  ) 

  print('\n------------------------------------\n Sample data table \n')
  sdata.sam.show()
  print('\n------------------------------------\n Cluster data table \n')
  sdata.clu.show()


if __name__ == "__main__":
  args = parse_args()
  start_spark()
  if not args.predownloaded:
    download_example(args.read_path)
  if args.scanpy_pca_clu:
    print("WARNING: using scanpy leiden clustering which requires you to install igraph yourself")
  main(args.read_path, args.write_path, args.no_save_bronze_silver, args.scanpy_pca_clu)
