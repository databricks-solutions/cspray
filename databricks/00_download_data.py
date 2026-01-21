# Databricks notebook source
# MAGIC %md
# MAGIC ## Ingest data from CELL X GENE into Unity Catalog
# MAGIC  - provide your catalog and schmea name in the the config.yaml
# MAGIC  - ensure the catalog and schema are already created
# MAGIC  - We will use all cores on all spark workers to do downloads in parallel
# MAGIC  - by default we download a subset of the data, but you could remove some of the filtering to download everything

# COMMAND ----------

# MAGIC %pip install 'gget>=0.25.7' cellxgene-census 'numpy<2' pybiomart scanpy pyyaml
# MAGIC %restart_python

# COMMAND ----------

import scanpy
import h5py

# COMMAND ----------

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
CATALOG = config.get("catalog", "")
SCHEMA = config.get("schema", "")
CENSUS_VERSION = config.get("census_version", "2023-01-30")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_h5ad")

# COMMAND ----------

SERVERLESS = False

if not SERVERLESS:
  CORES_PER_WORKER = 4 # YOULL NEED TO SET THIS
  NUMBER_OF_NODES = int(spark.conf.get('spark.databricks.clusterUsageTags.clusterWorkers'))

  CORES = CORES_PER_WORKER * NUMBER_OF_NODES
else:
  NUMBER_OF_NODES = 2
  CORES = 12

print(CORES)

# COMMAND ----------

import gget
import cellxgene_census
gget.setup("cellxgene")

df = gget.cellxgene(
    meta_only=True,
    census_version=CENSUS_VERSION,  
    species="homo_sapiens", 
    is_primary_data=True,
    suspension_type='cell',
    column_names=[
        'dataset_id',
        'assay',
        'cell_type',
        'donor_id'
    ]
)

sdf = spark.createDataFrame(df)

sdf_count = sdf.groupBy('dataset_id').count().sort('count', ascending=False)
sdf_count.write.format("delta").mode('overwrite').saveAsTable(f'{CATALOG}.{SCHEMA}.cellxgene_cell_counts')
sdf_count = spark.table(f'{CATALOG}.{SCHEMA}.cellxgene_cell_counts')


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

sdf_count = spark.table(f'{CATALOG}.{SCHEMA}.cellxgene_cell_counts')

dataset_id_sdf = sdf_count.filter(
    (F.col('count')<8_000)
    & (F.col('count')>6_000)
)

# COMMAND ----------

dataset_id_sdf.count()

# COMMAND ----------

import gget
import cellxgene_census
gget.setup("cellxgene")

if NUMBER_OF_NODES > 0:
  dataset_id_sdf = dataset_id_sdf.coalesce(1).repartition(CORES)

@F.udf(returnType=T.BooleanType())
def download_czi_hd5_to_volume(d_id):
    try:
        cellxgene_census.download_source_h5ad(
            dataset_id = d_id, 
            to_path = f'/Volumes/{CATALOG}/{SCHEMA}/raw_h5ad/{d_id}.h5ad', 
            census_version = CENSUS_VERSION,
            progress_bar=False,
        )
        output = True
    except:
        output = False
    return output

dataset_id_sdf = dataset_id_sdf.withColumn(
    'download_successful',
    download_czi_hd5_to_volume(F.col('dataset_id'))
)

dataset_id_sdf.write\
    .format('delta')\
    .mode("overwrite")\
    .option("overwriteSchema", "True") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.cellxgene_datasets_downloaded")

# COMMAND ----------

pdf = spark.table(f'{CATALOG}.{SCHEMA}.cellxgene_datasets_downloaded').orderBy('count', ascending=False).toPandas()
dataset_ids = list(pdf.dataset_id.values)

# COMMAND ----------

final_datasets = dict()

for d_id in dataset_ids:
    file_path = f'/Volumes/{CATALOG}/{SCHEMA}/raw_h5ad/{d_id}.h5ad'
    file = h5py.File(file_path, 'r')
    has_raw=False
    obs_in_raw = None
    if 'raw' in file:
        if (
            ('var' in file['raw'])
            and ('X' in file['raw'])
        ):
            obs_in_raw = 'obs' in file['raw']
            has_raw = True
            gene_count = file['raw']['var']['feature_name']['codes'].shape[0]
            entries = file['raw']['X']['indices'].shape[0]
            final_datasets[d_id] = {
                'gene_count': gene_count,
                'entries': entries,
                'obs_in_raw': obs_in_raw
            }
        else:
            print(f'missing var or X in raw for {d_id}')
    else:
        print(f'missing raw for {d_id}')

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(final_datasets).T
df = df.reset_index().rename(columns={'index':'dataset_id'})
df.display()
spark.createDataFrame(df).write\
    .format('delta')\
    .mode("overwrite")\
    .option("overwriteSchema", "True") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.cellxgene_datasets_processed")

# COMMAND ----------


