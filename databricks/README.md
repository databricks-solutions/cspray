### Downloading and processing data from CELLxGENE

The following notebooks will download data from CELLxGENE and process the raw data with cspray. Finally, we will perform reference-free cell annotation at the cluster level using majority voting over LLM calls, and merge these cell labels back into the cell-level data.

What you need to do:
 - Rename `config.yaml.example` to `config.yaml` and set the catalog and schema (ensure those catalog and schema exist).
 - Set up a compute cluster with some number of workers, let's say 4.
 - Run the 00 and 01 notebooks with this compute.
   - In the 01 notebook, you will want to change the WORKER_RAM : int = 32 to the RAM of your workers if it differs from 32GB.
 - Use serverless compute to run the 02 notebook.


Now, you'll download some data from CELLxGENE, process all the files simultaneously with cspray, and perform cell type labeling on the gold dataset.

The gold dataset is perfect for building dashboards and apps on top of it—allowing searching over cell types and other metadata to find samples of interest. Later, users can get the pre-processed data from the silver tables and perform aggregated analyses.

### A note on choosing compute

**Classic compute** is what we currently suggest for full end to end processing
(read through PCA, clustering and markers), and is why the 00 and 01 notebooks
run on a cluster. Intermediary persistence is on by default and nothing needs to
be disabled; the read arguments are sized from the cluster, as in the 01
notebook:

```
WORKER_RAM: int = 32   # RAM per worker, in GB

sdata = SprayData.from_h5ads(
    spark,
    path=path,
    force_partitioning=2*spark.sparkContext.defaultParallelism,
    chunk_size=int(6_000_000*(WORKER_RAM/16)),
)
```

**Serverless compute** does not support Spark `persist`/`cache`, so intermediary
persistence has to be turned off after constructing `sdata`:

```
sdata = SprayData.from_h5ads(
    spark,
    path=path,
    force_partitioning=500,
    chunk_size=int(6_000_000),
)
sdata.set_intermediary_persistance(persist=False)
```

With persistence off, processing works up to and including HVG. That covers
pipelines that only need QC, filtering and/or HVG. **PCA and clustering should be
run on classic compute for now**, which is why we suggest classic here for the
full pipeline. Defaults that make
serverless the easy path for more of the pipeline are in progress.

There is no fixed core count to size against on serverless, so partitioning is
set to a flat number rather than derived from `defaultParallelism`. The values
above are what we have found to work rather than package defaults - tune them to
your data and compute.

#### Ingesting already-processed files on serverless

The above is also all you need for the common case of ingesting a set of files
that have already been processed elsewhere - published atlases, or data a
collaborator has already filtered and normalised - where the goal is a queryable
set of tables rather than reprocessing. Read the files, land them as tables, then
sort out the metadata:

```
sdata = SprayData.from_h5ads(
    spark,
    path=path,
    from_raw=False,
    force_partitioning=500,
    chunk_size=int(6_000_000),
    obs_metadata_columns='all',
    var_metadata_columns='all',
)
sdata.set_intermediary_persistance(persist=False)

# write before touching the metadata: this is the only pass over the h5ad files
sdata.to_tables_and_reset(spark, table_base=f'{CATALOG}.{SCHEMA}', join_char='.bronze_')

# typed columns for the keys worth filtering on, with keys that are constant
# within a file collected onto the sample table instead of every cell
cs.md.promote_suggested(sdata, which='both', promote_sam=True)

# promotion only rewrites metadata, so X does not need writing a second time
# here we overwrite the tables with metadata, we could choose to construct new ones instead
sdata.to_tables_and_reset(
    spark,
    table_base=f'{CATALOG}.{SCHEMA}',
    join_char='.bronze_',
    subset=['obs', 'var', 'sam'],
)
```

Published files rarely agree on their obs columns, which is what the metadata
payload plus promotion is for: `cs.md.profile` reports what is actually in the
files, and `promote_suggested` materialises the useful keys as typed columns.
`promote_sam=True` routes sample-level values such as tissue, assay or disease to
`sam` (one row per input file) rather than repeating them on every cell, so the
sample table becomes the natural place to browse and filter what has been
ingested. Sample annotations you hold outside the files can be joined on with
`cs.md.add_sample_metadata(sdata, metadata_df, on='file_path')`.

