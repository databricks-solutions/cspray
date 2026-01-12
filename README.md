
<div align="center">
  <img src="static/logo_cspray.png" alt="Cspray Logo" width="25%"/>
  
  # cspray 
  ### Cell SPRAY: Distributed single cell analysis with pyspark
  #### pronounced "sea spray"; /ˈsiː spreɪ/
  
  <!-- [![License](https://img.shields.io/badge/license-blue.svg)](LICENSE) -->
  <!-- [![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/) -->
  <!-- [![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)](https://spark.apache.org/) -->
  
  **Scale your single-cell RNA analysis across atlas-sized datasets**
  
  [Installation](#install) • [Quick Start](#minimal-example) • [About](#about)
  
</div>

---

## 🚀 Why Cspray?

- **🌐 Truly Distributed**: Process multiple atlas-scale h5ad files without RAM limitations
- **⚡ Spark-Native**: Leverages PySpark for horizontal scaling across compute clusters
- **🔄 Familiar API**: Similar to scanpy/AnnData for easy adoption
- **💾 Smart I/O**: Chunked reading means file size doesn't dictate compute requirements
- **📊 Delta Lake Ready**: Native support for Delta tables and incremental processing

## Install

Current install method is to clone the repo and pip install via path:
```
~: git clone the/cspray/url
~: pip install cspray
```

An end to end example for use on a local machine (with at least 12Gb RAM) is included under examples/. This will download a small public dataset, ingest and process that dataset and print some details and final processed tables to terminal. This example will start a spark session for you, but you will need to additionally install java for spark to work if it is not already present in your environment.

```
~: cd examples
~: python examples/single_run_example.py -r tmp_sample.h5ad -w tmp_delta/
```
to see all options:
```
~: python examples/single_run_example.py --help
```

Additionally for Databricks users there are some example notebooks and we expect to extend these offerings.

## About

CSpray is a python package for end-to-end processing of single cell data using pyspark for efficient distributed compute. This allows for scaling of single cell RNA analysis over the scale of many samples, including multiple atlas scale h5ad files, and with increasing sample sizes as technology continues to improve. The package has several key modules that represent key analysis stages:
 
 - read : ingestion of h5ad to dataframes
 - preprocessing : gene/cell filtering, highly variable gene annotation
 - tools : PCA, clustering, marker gene extraction

In Cspray, the main storage element for the single cell dataset (which can include many samples) is the "SprayData" object. This object is similar in some ways to the [AnnData](https://anndata.readthedocs.io/en/stable/) object, but the elements of the data object (X, obs, var, etc.) are stored as pyspark dataframes instead of as pandas dataframes and sparse arrays. The behavior of pyspark dataframes makes the interaction similar to interacting with AnnData objects backed to disk, i.e., the data is not in RAM, but the actual processing can also be distributed.

Reading of h5ad files is performed in a chunked manner, and this means we do not need to consider the sizes of individual files when choosing the RAM of worker nodes. The chunk size that is required is determined only by the RAM of the workers and not the files we wish to read. This means that reading a large number of samples with a large skew in file sizes does not require any tuning of compute sizes to handle this variation or to limit the cost of using very large compute to process smaller files just because, say, one file was large. This issue of how to select the compute size for analysis of single cell data has long been a problem, but by using fixed chunk sizing, the size of worker nodes required is known ahead of time. For instance, we find that nodes of 16Gb RAM are perfectly acceptable, whereas for many h5ad files we ingest, the RAM required to read the h5ad file to memory would be significantly larger.

Data can be written by spark to disk according to the users preference, parquet and Delta format files being standard choices. The Delta format offers many useful benefits for data organization, compression, and many other features that make it great at things like data incrementalization and Change Data Capture. See [here](https://delta.io/blog/open-table-formats/) for more details on open table formats such as Delta.

All stages are written in native PySpark, or with convenient PySpark tooling such as mapInArrow, or PySparkML, allowing us to distribute across machines. Until the HVG stage, the processing matches the default Scanpy operations. For the tooling stage, we have two options - using PCA and KMeans for clustering using PySparkML libraries, or using Scanpy PCA and Leiden clustering via a mapInArrow operation. The latter does require shuffling all sample data to individual workers, which, though the prior HVG step does lead to data reduction, can still cause out-of-memory issues for very large samples. We therefore suggest using the default PySparkML implementations for PCA and clustering. While KMeans is not an ideal method for scRNA clustering in general, we view this as a reasonable choice for large-scale data processing. This is because one can always pick up individual samples at any prior stage (post-filtering, post-HVG) for more niche analysis, and for end-to-end processing at scale, we wish only to get high-level clusters for the sake of data discovery over all samples. For instance, we may want to only generate clusters of fairly major cell types, e.g., T cells, B cells, rather than specific subsets. When analyzing very large numbers of samples at scale, the pre-processing is not tuned per sample, and so one would not necessarily expect the data to be perfectly filtered and prepared for clustering (or clustered at the right resolution) for good cell labeling in any case.

Ultimately, with high-level clusters of cells identified across many samples, one can perform reference-free annotation of each cluster. Since this process can rely on the marker genes already calculated per cluster and an LLM, the total number of LLM calls required is relatively few and therefore makes for a cost-effective approach for performing cell type labeling at large scale for the purpose of data discovery. Say a user wants to find all cells of a certain type in samples from certain tissues, they'll be able to find these quickly. They can then pull the data from all samples post-PCA into a SprayData object and continue their analysis from that point with the pre-filtering and HVG already performed.

## Minimal Example

```
from cspray.data import SprayData
import cspray as cs

path = 'sample.h5ad' # single file path, list of paths, or dir path
ensembl_reference_df = cs.utils.get_gene_table()
sdata = SprayData.from_h5ads(
    spark,
    path=path,
    ensembl_reference_df=ensembl_reference_df
)
sdata.to_tables_and_reset(spark, table_base='singlecell.bronze')

cs.pp.calculate_qc_metrics(sdata)
cs.pp.filter_cells(sdata)
cs.pp.filter_genes(sdata)
cs.pp.apply_samplewise_mt_statistic(sdata)
cs.pp.filter_cells_on_mt(sdata)
cs.pp.normalize(sdata)
cs.pp.log1p_counts(sdata)
sdata.to_tables_and_reset(spark, table_base='singlecell.silver', join_char='.pp_')

cs.pp.calculate_hvg(sdata, n_hvg=500)
sdata.to_tables_and_reset(spark,table_base='singlecell.silver', join_char='.hvg_') 

pp.pca(sdata)
sdata.to_tables_and_reset(spark,table_base='singlecell.silver', join_char='.pca_', subset=['obs'])

scores_pdf = cs.tl.kmeans(sdata, ks=[2,3])
sdf_rank = cs.tl.rank_marker_genes(sdata, fc_cutoff=0.15)
sdata.to_tables_and_reset(spark,table_base='singlecell.silver', join_char='.end_')

cs.tl.as_gold_mart_data(sdata)
sdata.to_tables_and_reset(spark,table_base='singlecell.gold', join_char='.') 
```



### Run tests
From the root dir, pip install the test environment and run tests: 
```
    ~$: pip install .[test]
    ~$: sh run_tests.sh
```

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.


## Dependencies

### Core Packages

| Package | License | Source |
|---------|---------|--------|
| anndata | BSD-3-Clause | https://github.com/scverse/anndata |
| scanpy | BSD-3-Clause | https://github.com/scverse/scanpy |
| bbknn | MIT | https://github.com/Teichlab/bbknn |
| pyspark | Apache-2.0 | https://github.com/apache/spark |
| pyarrow | Apache-2.0 | https://github.com/apache/arrow |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy |
| scipy | BSD-3-Clause | https://github.com/scipy/scipy |
| pooch | BSD-3-Clause | https://github.com/fatiando/pooch |
| pybiomart | MIT | https://github.com/jrderuiter/pybiomart |

### additional: Cell Type Annotation

| Package | License | Source |
|-------|---------|--------|
| openai (library) | Apache-2.0 | https://github.com/openai/openai-python |

*Note: AI models are used in the annotation workflow for automated cell type labeling based on marker genes.*

### additional: Benchmark and Databricks Examples Packages

| Package | License | Source |
|---------|---------|--------|
| gget | BSD-2 | https://github.com/pachterlab/gget |
| cellxgene-census | MIT | https://github.com/chanzuckerberg/cellxgene-census |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| h5py | BSD-3-Clause | https://github.com/h5py/h5py |
| databricks-sdk | Apache-2.0 | https://github.com/databricks/databricks-sdk-py |
| pyyaml | MIT | https://github.com/yaml/pyyaml |