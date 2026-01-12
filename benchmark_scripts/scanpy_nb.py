# Databricks notebook source
# MAGIC %pip install 'numpy<2' scanpy==1.11.4 anndata 
# MAGIC %pip install pybiomart
# MAGIC ### %pip install igraph # we will leave this off by default, but you need to uncomment if you wish to use Leiden clustering
# MAGIC %restart_python

# COMMAND ----------

import scanpy as sc
import anndata as ad
import os
import pandas as pd
import numpy as np
import time
import shutil
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

import os

def get_gene_table(species='hsapiens'):
    """
    Download gene reference table from Ensembl.
    
    Args:
        species: Species identifier (e.g., 'hsapiens', 'mmusculus', 'rnorvegicus')
    
    Returns:
        DataFrame with ensembl_gene_id and external_gene_name columns
    """
    from pybiomart import Dataset
    print(f"Querying Ensembl for {species}...")
    dataset = Dataset(name=f'{species}_gene_ensembl',
                    host='http://www.ensembl.org')
    
    gene_table = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    
    # Rename BioMart's human-readable column names to code-friendly names
    gene_table.rename(columns={
        'Gene stable ID': 'gene_id',
        'Gene name': 'gene_name'
    }, inplace=True)
    
    print(f"Retrieved {len(gene_table)} genes for {species}")

    return gene_table

# COMMAND ----------

dbutils.widgets.text("sample_path", "", "Sample Path")
dbutils.widgets.text("clustering_method", "", "kmeans")
dbutils.widgets.text("out_dir", "", "")

# COMMAND ----------

path = dbutils.widgets.get("sample_path")
CLUSTERING = dbutils.widgets.get("clustering_method")
OUTDIR = dbutils.widgets.get("out_dir")
print(path)
print(CLUSTERING)
print(OUTDIR)

# COMMAND ----------

sc.settings.n_jobs = -1

# COMMAND ----------

ALLOWED_CLUSTERING = ["leiden","kmeans"]
if CLUSTERING not in ALLOWED_CLUSTERING:
    raise ValueError(f"Unknown clustering method {CLUSTERING}")

# COMMAND ----------

GENE_NAME_COLUMN = 'index'

# COMMAND ----------

import h5py
with h5py.File(path, 'r') as f:
    groups = list(f.keys())
    
if 'raw' not in groups:
    print("no raw in here")

# COMMAND ----------

gene_table = get_gene_table()

# COMMAND ----------

t0 = time.time()
adata = sc.read_h5ad(
    path,
)
adata.obs_names_make_unique()

# here we use adata.raw if it exists as is assumeed to be unprocessed raw data
if adata.raw is not None:
    adata = adata.raw.to_adata()

GENE_NAME_COLUMN = adata.var.index.name
if GENE_NAME_COLUMN is None:
    GENE_NAME_COLUMN = 'index'

adata.var = adata.var.reset_index()
adata.var[GENE_NAME_COLUMN] = adata.var[GENE_NAME_COLUMN].astype(str) 
# adata.var['Gene name'] = adata.var[GENE_NAME_COLUMN]
# adata.var = adata.var.set_index(GENE_NAME_COLUMN)

adata.var = adata.var.rename(columns={'gene_name':'original_gene_name'})
adata.var = adata.var.merge(
    gene_table,
    left_on=GENE_NAME_COLUMN,
    right_on='gene_id',
    how='left'
)
adata.var = adata.var.set_index(GENE_NAME_COLUMN)

adata.var["mt"] = adata.var['gene_name'].str.startswith("MT-",na=False)
# ribosomal genes
# adata.var["ribo"] = adata.var['Gene name'].str.startswith(("RPS", "RPL"),na=False)
# hemoglobin genes
# adata.var["hb"] = adata.var['Gene name'].str.contains("^HB[^(P)]",na=False)

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], inplace=True, log1p=True
)

print("start filter")
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 8, :]

sc.pp.normalize_total(adata,target_sum=1e4)
sc.pp.log1p(adata)

print("filtered and normed")

sc.pp.highly_variable_genes(adata, n_top_genes=1000)

print("HVG")

sc.tl.pca(adata, mask_var="highly_variable")

print("PCA")

if CLUSTERING == 'leiden':
    # remove any existing neighbors state that could cause issues
    if 'neighbors' in adata.uns:
        adata.uns.pop('neighbors')
    
    sc.pp.neighbors(adata)
    sc.tl.leiden(
        adata,
        key_added='cluster',
        resolution=0.05,
        random_state=0,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
elif CLUSTERING == 'kmeans':

    best_score = 0
    for cl in [2,3,4,5]:
        kmeans = MiniBatchKMeans(n_clusters=cl, random_state=0, n_init="auto", max_iter=20)
        print("doing kmean ,,  ",cl)
        kmeans.fit_predict(adata.obsm['X_pca'])
        print("fit done...")
        labels = kmeans.labels_
        score = silhouette_score(adata.obsm['X_pca'], labels, metric='euclidean')
        if score>best_score:
            best_score = score
            best_labels = labels
            best_cl = cl

    adata.obs['cluster'] = pd.Categorical(
        values=best_labels.astype("U"),
        categories=sorted(map(str, np.unique(best_labels))),
    )
else:
    raise ValueError(f"Unknown clustering method {CLUSTERING}, please select from {ALLOWED_CLUSTERING}")

print("Clustered")
sc.tl.rank_genes_groups(adata, "cluster") # use default = t test , method="wilcoxon")
marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(10)
display(marker_df)

# to local store with the path filename adjusted for clustering
outfile = os.path.join('/local_disk0', os.path.basename(path).replace('.h5ad', f'_{CLUSTERING}.h5ad'))

def columns_clean(df):
    # sanitizes columns, slashes remove etc
    df.columns = (
        df.columns.str.replace(r"[\/\\\s]", "_", regex=True)
                  .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df

adata.obs = columns_clean(adata.obs)
adata.var = columns_clean(adata.var)

adata.write_h5ad(outfile)


t1 = time.time()

# now cp to Volumes (shutil to OUTDIR variable)
# we don't include that in the runtime
shutil.copy(outfile, OUTDIR)

# COMMAND ----------

# set t1-10 as a value in notebook values for output
dbutils.jobs.taskValues.set(key="execution_time", value= t1 - t0)

ex_time = t1-t0
dbutils.notebook.exit("{:.3f}".format(ex_time))

# COMMAND ----------


