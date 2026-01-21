# Databricks notebook source
# MAGIC %pip install openai
# MAGIC %pip install anndata
# MAGIC %restart_python

# COMMAND ----------

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
CATALOG = config.get("catalog", "")
SCHEMA = config.get("schema", "")

# COMMAND ----------

from openai import OpenAI
import os
import pandas as pd

# or your key
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# or your url
BASE_URL = f"{dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()}/serving-endpoints"

client = OpenAI(
  api_key=DATABRICKS_TOKEN, 
  base_url=BASE_URL, 
)


# COMMAND ----------

prompt = """
You will be given a list of genes that are most differentially expressed in a cluster of cells in a scRNA analysis as well as a tissue type the cells are from. 
Your task is to give the cell type label from the list below. You MUST only provide your final answer and no further information. e.g "glial cell" or "fibroblast" NEVER any thoughts or discussion

Assign a cell type from one of these:
['T cell', 'endothelial cell of vascular tree', 'retinal cell', 'visual system neuron', 'ecto-epithelial cell', 'GABAergic interneuron', 'neurecto-epithelial cell', 'stem cell', 'macrophage', 'epithelial cell of nephron', 'sensory neuron', 'connective tissue cell', 'mononuclear phagocyte', 'kidney medulla cell', 'smooth muscle cell', 'neuron associated cell', 'stromal cell', 'columnar/cuboidal epithelial cell', 'blood vessel endothelial cell', 'mature B cell', 'endo-epithelial cell', 'professional antigen presenting cell', 'cerebral cortex neuron', 'meso-epithelial cell', 'neuron of the forebrain', 'fibroblast', 'myeloid cell', 'kidney epithelial cell', 'afferent neuron', 'kidney cell', None, 'striated muscle cell', 'epithelial cell of alimentary canal', 'retinal ganglion cell', 'lymphocyte of B lineage', 'interneuron', 'macroglial cell', 'intestinal epithelial cell', 'muscle cell', 'hematopoietic lineage restricted progenitor cell', 'myeloid leukocyte', 'inhibitory interneuron', 'glial cell', 'glutamatergic neuron', 'visceral muscle cell', 'phagocyte (sensu Vertebrata)', 'B cell', 'endothelial cell', 'dendritic cell', 'alpha-beta T cell', 'non-striated muscle cell', 'glandular secretory epithelial cell', 'endocrine cell', 'GABAergic neuron', 'progenitor cell', 'kidney cortical cell', 'conventional dendritic cell', 'peripheral nervous system neuron', 'secretory epithelial cell', 'autonomic neuron', 'efferent neuron', 'cardiocyte', 'sensory receptor cell', 'lymphocyte of B lineage, CD19-positive', 'mature T cell']
"""

# COMMAND ----------

from typing import List
def get_cell_type(genes : List[str], tissue : str = 'unknown', model : str = "databricks-claude-sonnet-4") -> str:
    chat_completion = client.chat.completions.create(
    messages=[
        {
        "role": "system",
        "content": prompt,
        },
        {
        "role": "user",
        "content": f"Tissue={tissue},  list of marker genes = {str(genes)}",
        }
    ],
    model=model,
    max_tokens=256
    )
    content = chat_completion.choices[0].message.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return content[-1]['text']
    else:
        return content

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examples in general

# COMMAND ----------

genes0 = ["CCL4","IL1B","IFIT2","CCL3","CCL20","TNFAIP3","CXCL3","IFIT3","CXCL8","CCL3L3"]
genes1 = ["IGHV1-3","IGKV1-39","IGHV2-5","IGHV3-23","IGLL5","IGHV5-51","IGHV3-7","IGLV2-14","MT3","BCAS1"]


print(get_cell_type(genes0))
print(get_cell_type(genes1))

# COMMAND ----------

print(get_cell_type(genes1, model='databricks-gpt-5-1', tissue='blood'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data

# COMMAND ----------

clu_path = f'{CATALOG}.{SCHEMA}.gold_clu'
pdf = spark.table(clu_path).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Collect the tissue information from the raw h5ad
# MAGIC  - note: we could include this during the cspray processing, but we did not in the benchmark run

# COMMAND ----------

import anndata as ad
tissue_map = dict()
for fp in pdf.file_path.unique():
    sample_idx = fp.split('/')[-1].split('.')[0]
    scanpy_clustering = fp.split('/')[-1].split('.')

    f = f'/Volumes/{CATALOG}/{SCHEMA}/raw_h5ad/{sample_idx}.h5ad'
    adata = ad.read_h5ad(f, backed='r')
    print(fp, adata.obs['tissue'].mode().iloc[0])
    tissue_map[fp] = adata.obs['tissue'].mode().iloc[0]

pdf['tissue'] = pdf['file_path'].map(tissue_map)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply the cell labeling
# MAGIC - note that ai_query can be used to run this at very large scale too and is much more efficient as requests can be placed in parallel

# COMMAND ----------

pdf['cell_label'] = pdf.apply(lambda row: get_cell_type(row['marker_genes'], tissue=row['tissue'], model='databricks-gpt-5-1'), axis=1)

for i in range(5):
  pdf[f'cell_label_{i}'] = pdf.apply(lambda row: get_cell_type(row['marker_genes'], tissue=row['tissue'], model='databricks-gpt-5-1'), axis=1)

# COMMAND ----------

pdf.display()

# COMMAND ----------

# take the modal cell type among cell_label_i for i 0-4 as cell_label
pdf['cell_label'] = pdf[[f'cell_label_{i}' for i in range(5)]].mode(axis=1)[0]

# COMMAND ----------

pdf.display()

# COMMAND ----------

spark.createDataFrame(
    pdf[['cluster_id','file_path','cell_label']]
).write\
 .mode('overwrite')\
 .saveAsTable(f'{CATALOG}.{SCHEMA}.cluster_annotation')

# COMMAND ----------

# MAGIC %md
# MAGIC ## now join back into obs
# MAGIC  - as part of pipeline could than have cell type annotations in clu and obs tables
# MAGIC  - this could be published in dashboards/apps for discovery
# MAGIC  - can analyze overlap if there are known labels

# COMMAND ----------

sdf_obs = spark.table(f"{CATALOG}.{SCHEMA}.gold_obs")
sdf_obs = sdf_obs.join(
    spark.table(f'{CATALOG}.{SCHEMA}.cluster_annotation').withColumnRenamed('cell_label','cspray_cell_label') , 
    on=['cluster_id','file_path'], 
    how='left'
)

# COMMAND ----------

sdf_obs.display()

# COMMAND ----------

# could merge instead, but for simple demo, overwrite
sdf_obs.write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable(f'{CATALOG}.{SCHEMA}.gold_obs')

# COMMAND ----------


