# Databricks notebook source
# MAGIC %pip install git+https://github.com/databricks-solutions/cspray.git@main 
# MAGIC %restart_python

# COMMAND ----------

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
CATALOG = config.get("catalog", "")
SCHEMA = config.get("schema", "")


# COMMAND ----------

from cspray.data import SprayData
import cspray as cs
import pandas as pd

# (optionally) prevent mlflow logging of PCA and kmeans
import mlflow
mlflow.autolog(disable=True)

# RAM on each worker (only required for read stage -use it to help set read in chunk size optimally)
WORKER_RAM : int = 32
TOTAL_CORES = spark.sparkContext.defaultParallelism
NUM_WORKERS = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1
CORES_PER_WORKER = TOTAL_CORES // NUM_WORKERS
MAX_PARTITIONS = 200

BASE_prefix = "cspray"

# COMMAND ----------

datasets = list(spark.table(f"{CATALOG}.{SCHEMA}.cellxgene_datasets_processed").toPandas()['dataset_id'].values)
root_path = f"/Volumes/{CATALOG}/{SCHEMA}/raw_h5ad/"

path = [
    root_path + d + '.h5ad' for d in datasets
]

ensembl_reference_df = spark.createDataFrame(cs.utils.get_gene_table())

# COMMAND ----------

import time

t0 = time.time()

sdata = SprayData.from_h5ads(
    spark,
    path=path, 
    force_partitioning = 2*spark.sparkContext.defaultParallelism,
    chunk_size=int(6_000_000*(WORKER_RAM/16)),
    from_raw=True,
    fallback_default=False,
    broadcast_genes=True,
    ensembl_reference_df=ensembl_reference_df,
)
sdata.to_tables_and_reset(spark, table_base=f'{CATALOG}.{SCHEMA}', join_char=f'.{BASE_prefix}_bronze_')

cs.pp.calculate_qc_metrics(sdata)
cs.pp.filter_cells(sdata)
cs.pp.filter_genes(sdata)
cs.pp.apply_samplewise_mt_statistic(sdata)
cs.pp.filter_cells_on_mt(sdata)
cs.pp.normalize(sdata)
cs.pp.log1p_counts(sdata)
sdata.to_tables_and_reset(spark, table_base=f'{CATALOG}.{SCHEMA}', join_char=f'.{BASE_prefix}_silver_pp_')

cs.pp.calculate_hvg(sdata, n_hvg=1000)
sdata.to_tables_and_reset(spark,table_base=f'{CATALOG}.{SCHEMA}', join_char=f'.{BASE_prefix}_silver_hvg_') 

cs.pp.pca(sdata)
sdata.to_tables_and_reset(spark,table_base=f'{CATALOG}.{SCHEMA}', join_char=f'.{BASE_prefix}_silver_pca_', subset=['obs'])

scores_pdf = cs.tl.kmeans(sdata, ks=[2,3,4,5])
sdf_rank = cs.tl.rank_marker_genes(sdata, fc_cutoff=0.15)
sdata.to_tables_and_reset(spark,table_base=f'{CATALOG}.{SCHEMA}', join_char=f'.{BASE_prefix}_silver_end_')


cs.tl.as_gold_mart_data(sdata)
sdata.to_tables_and_reset(spark,table_base=f'{CATALOG}.{SCHEMA}', join_char='.gold_') 

t1 = time.time()
print(f"Total time: {(t1-t0)} seconds")
print(f"Total time: {(t1-t0)/60.} minutes")
print(f"Total time: {(t1-t0)/60./60.} hours")
