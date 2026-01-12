"""
Module for single-cell RNA-seq data processing and quality control in Spark.
Provides functions for PCA, gene/cell filtering, normalization, log transformation,
highly variable gene selection, and sample/cell-level QC metrics.
"""
from .data import SprayData
from typing import Optional, List
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import Bucketizer
import numpy as np

from .utils import materialize
from . import _pca

def pca(sdata: SprayData, n_hvg:Optional[int]=None, n_components:Optional[int]=50):
    """
    Perform principal component analysis (PCA) on the single-cell RNA-seq data.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    n_hvg : int, optional
        Number of highly variable genes to use for PCA. Defaults to 1000 if not specified.
    n_components : int, optional
        Number of principal components to compute. Defaults to 50 if not specified.

    Returns
    -------
    None
        Updates sdata.obs with PCA results.
    """
    if n_hvg is None:
        n_hvg = 1000
        print('Warning: using a fixed default 1000 hvg - if you used something else change it!')
    else:
        print(f'using the passed n_hvg: {n_hvg}')
    pca_df = _pca.pca(sdata,n_hvg,n_components)
    sdata.obs = sdata.obs.join(
        pca_df,
        on=['fp_int','cell_idx'],
        how='left'
    )
    # note that gene expression map remains in obs - good for downstream
    # marker gene annotation but wouldn't want it saved in obs
    return None

def make_var_names_unique(sdata, remove_null_genes=True):
    """
    Ensure gene_names are unique within each sample.
    Sums expression values for duplicate gene_names.
    """
    if remove_null_genes:
        new_var = sdata.var.filter(F.col('gene_name').isNotNull())
    else:
        new_var = sdata.var
    
    # Check if deduplication needed
    dup_count = (
        new_var
        .groupBy('fp_int', 'gene_name')
        .count()
        .filter(F.col('count') > 1)
        .count()
    )
    
    if dup_count == 0:
        print("No duplicate gene_names found")
        return
    
    print(f"Deduplicating gene_names...")
    
    # Deduplicate var
    new_var = (
        new_var
        .groupBy('fp_int', 'gene_name')
        .agg(
            F.min('gene_idx').alias('gene_idx'),
            F.first('gene_id').alias('gene_id'), # exact mapping could change here but its not so important 
            F.first('fp_int').alias('fp_int'),
            F.concat_ws('|', F.collect_set('original_provided_gene_name')).alias('original_provided_gene_name')
        )
    ).repartition(sdata.var.rdd.getNumPartitions(), 'fp_int')
    
    # Add file_path back (lost in groupBy at line 77)
    new_var = new_var.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')
    
    if sdata.persist_intermediaries:
        new_var.persist(sdata.persist_storage_level)

    # Update and aggregate X
    sdata.X = (
        sdata.X
        .drop('gene_idx')
        .join(
            new_var.select('fp_int', 'file_path', 'gene_name', 'gene_idx'),
            on=['fp_int', 'file_path', 'gene_name']
        )
        .groupBy('fp_int', 'cell_idx', 'gene_idx')
        .agg(
            F.sum('expression').alias('expression'),
        )
    ).repartition(sdata.X.rdd.getNumPartitions(), 'fp_int')
    return None

# def make_var_names_unique(sdata):
#     duplicate_genes = (
#         sdata.var
#         .groupBy('file_path', 'gene_name')
#         .agg(F.count('*').alias('dup_count'))
#         .filter(F.col('dup_count') > 1)
#         .select('file_path', 'gene_name')
#     )

#     var_dedup = (
#         sdata.var
#         .groupBy('file_path', 'gene_name')
#         .agg(
#             F.first('gene_idx').alias('gene_idx'), 
#             F.first('gene_id').alias('gene_id'), 
#             F.first('fp_int').alias('fp_int'),
#             F.first('original_provided_gene_name').alias('original_provided_gene_name')
#         )
#     )
    
#     X_with_issues = sdata.X.join(
#         duplicate_genes,
#         on=['file_path', 'gene_name'],
#         how='inner'  # Only rows with duplicate gene_names
#     )

#     X_clean = sdata.X.join(
#         duplicate_genes,
#         on=['file_path', 'gene_name'],
#         how='left_anti'  # All rows WITHOUT duplicate gene_names
#     )

#     X_fixed = X_with_issues.groupBy(
#         'file_path', 'cell_idx', 'gene_name'
#     ).agg(
#         F.sum('expression').alias('expression'),
#         F.first('gene_idx').alias('gene_idx'),
#         F.first('fp_int').alias('fp_int'),
#     )

#     #  Union back together
#     sdata.X = X_clean.union(X_fixed)
#     sdata.var = var_dedup

def filter_missing_genes(sdata: SprayData):
    """
    Filter out genes with missing gene_name from sdata.var and sdata.X.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.

    Returns
    -------
    None
        Updates sdata.var and sdata.X by removing rows with missing gene_name.
    """
    sdata.var = sdata.var.filter(F.col('gene_name').isNotNull())
    sdata.X = sdata.X.filter(F.col('gene_name').isNotNull())
    return None

def calculate_qc_metrics(sdata: SprayData, mt_prepend='MT-'):
    """
    Calculate quality control (QC) metrics for single-cell RNA-seq data.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.

    Returns
    -------
    None
        Updates sdata.X, sdata.obs, sdata.var, and sdata.sam with QC metrics such as total counts,
        percent mitochondrial reads, number of detected genes per cell, and number of cells per gene.
    """

    count_df =  sdata.X.groupby('fp_int', 'cell_idx').agg(F.sum('expression').alias('total_counts'))

    # n mitochondrial reads
    sdata.X =  sdata.X.withColumn('is_mt', F.col('gene_name').startswith(mt_prepend))
    mt_count_df =  sdata.X.filter(F.col('is_mt')).groupby('fp_int', 'cell_idx').agg(F.sum('expression').alias('mt_counts'))


    # get the pcnt of reads that are mitochondrial in each sample,cell group
    pct_mt_reads_df = mt_count_df.join(
        count_df, 
        ['fp_int', 'cell_idx'],
        how='right'
        )
    pct_mt_reads_df = pct_mt_reads_df.withColumn('mt_counts', F.when(F.col('mt_counts').isNull(), 0).otherwise(F.col('mt_counts')))
    pct_mt_reads_df = pct_mt_reads_df.withColumn('pct_mt_reads', (F.col('mt_counts') / F.col('total_counts')) * 100)

    sdata.X =  sdata.X\
        .join(
            count_df, 
            ['fp_int', 'cell_idx'], 
            'inner')\
        .join(
            pct_mt_reads_df.select('fp_int', 'cell_idx', F.col('pct_mt_reads').alias('pct_mt')), 
            ['fp_int', 'cell_idx'], 
            'left' # mt_count could be missing cells if no mt reads in a cell, so do left join, some pct_mt COULD be NULL in principal
        )

    # number of cells that have >0 reads per gene
    cell_bincount_df = sdata.X.groupby('fp_int', 'gene_idx').agg(F.sum((F.col('expression') > 0).cast('int')).alias('bin_cell_counts'))
    # number of genes that have >0 reads per cell
    gene_bincount_df = sdata.X.groupby('fp_int', 'cell_idx').agg(F.sum((F.col('expression') > 0).cast('int')).alias('bin_gene_counts'))

    

    sdata.X = sdata.X\
    .join(
        cell_bincount_df, 
        ['fp_int', 'gene_idx'], 
        'inner')\
    .join(
        gene_bincount_df,
        ['fp_int','cell_idx'],
        'inner'
    )

    # cell wise
    cell_sdf = sdata.X.groupby('fp_int', 'cell_idx').agg(
        F.first('total_counts').alias('total_count'),
        F.first('bin_gene_counts').alias('n_genes_by_counts'),
        F.first('pct_mt').alias('pct_mt'),
    )

    if sdata.persist_intermediaries:
        count_df.persist(sdata.persist_storage_level)
        cell_bincount_df.persist(sdata.persist_storage_level)
        gene_bincount_df.persist(sdata.persist_storage_level)
        cell_sdf.persist(sdata.persist_storage_level)
        if sdata.materialize_off_route:
            materialize(cell_sdf)


    sdata.sam = count_df.groupby('fp_int').agg(
        F.count('cell_idx').alias('n_cells'),
        F.sum('total_counts').alias('total_counts')
    )
    #  and mean genes per cell
    sdata.sam = sdata.sam.join(
        gene_bincount_df.groupby('fp_int').agg(
            F.mean(F.col('bin_gene_counts')).alias('mean_genes_per_cell')
        ),
        on='fp_int',
        how='left'
    )
    
    # Add file_path to sam (created via groupBy, so only has fp_int)
    sdata.sam = sdata.sam.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')

    
    sdata.obs = sdata.obs.join(
        cell_sdf,
        on=['fp_int', 'cell_idx'],
        how='left'
    ) 

    sdata.var = sdata.var.join(
        cell_bincount_df,
        on=['fp_int','gene_idx'],
        how='left'
    )

    return None

# SHOULD I ALSO FILTER OBS AND VAR DURING FILTERING?

def filter_cells(sdata: SprayData, min_genes:Optional[int]=100):
    """
    Filter out cells with fewer than min_genes detected genes.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    min_genes : int, optional
        Minimum number of detected genes required for a cell to be retained. Defaults to 100.

    Returns
    -------
    None
        Updates sdata.X and sdata.obs by removing cells with insufficient detected genes.
    """
    sdata.X = sdata.X.filter(
        F.col('bin_gene_counts') >= min_genes
    )
    sdata.obs = sdata.obs.filter(
        F.col('n_genes_by_counts') >= min_genes
    )
    return None

def filter_genes(sdata: SprayData, min_cells:Optional[int]=3):
    """
    Filter out genes detected in fewer than min_cells cells.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    min_cells : int, optional
        Minimum number of cells a gene must be detected in to be retained. Defaults to 3.

    Returns
    -------
    None
        Updates sdata.X and sdata.var by removing genes detected in too few cells.
    """
    sdata.X = sdata.X.filter(
        F.col('bin_cell_counts') >=  min_cells
    )
    sdata.var = sdata.var.filter(
        F.col('bin_cell_counts') >=  min_cells
    )
    return None

def apply_samplewise_mt_statistic(sdata: SprayData,thresholds:Optional[List[float]]=None):
    """
    Calculate, for each sample, the fraction of cells passing various mitochondrial percentage thresholds.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    thresholds : list of float, optional
        List of mitochondrial percentage thresholds to evaluate. Defaults to [2, 5, 8, 10].

    Returns
    -------
    None
        Updates sdata.sam with columns indicating the fraction of cells passing each mitochondrial threshold.
    """
    if thresholds is None:
        thresholds = [2,5,8,10]
    
    for thr in thresholds:
        sdata.sam = sdata.sam.join(
            sdata.obs.select('fp_int','pct_mt').groupby('fp_int').agg(
                (
                    F.sum(F.when(F.col('pct_mt') < thr, 1).otherwise(0)) / F.count('*'))\
                    .alias('pct_cells_passing_mt_{:.1f}_pct'.format(thr)
                )
            ),
            on='fp_int',
            how='left'
        )

def filter_cells_on_mt(sdata: SprayData, max_mt_pct:Optional[float]=8):
    """
    Filter out cells with mitochondrial percentage greater than or equal to max_mt_pct.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    max_mt_pct : float, optional
        Maximum allowed percentage of mitochondrial reads per cell. Cells with pct_mt >= max_mt_pct are removed. Defaults to 8.

    Returns
    -------
    None
        Updates sdata.X and sdata.obs by removing cells exceeding the mitochondrial threshold.
    """
    # if also filter obs - filter both OR filter obs and join?
    sdata.X = sdata.X.filter(
        F.col('pct_mt') < max_mt_pct
    )
    sdata.obs = sdata.obs.filter(
        F.col('pct_mt') < max_mt_pct
    )

def normalize(sdata: SprayData, target:Optional[int]=10_000):
    """
    Normalize gene expression counts per cell to a target sum.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    target : int, optional
        Target sum of normalized counts per cell. Defaults to 10,000.

    Returns
    -------
    None
        Updates sdata.X with a new column 'norm_counts' containing normalized expression values.
    """
    sdata.X = sdata.X.withColumn(
        'norm_counts', 
        F.col('expression')*(target/F.col('total_counts'))
    )

def log1p_counts(sdata: SprayData):
    """
    Apply log1p transformation to normalized gene expression counts.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.

    Returns
    -------
    None
        Updates sdata.X with a new column 'log1p_norm_counts' containing log1p-transformed normalized counts.
    """
    sdata.X = sdata.X.withColumn(
        'log1p_norm_counts', 
        F.log1p(F.col('norm_counts'))
    )

def calculate_hvg(
    sdata: SprayData,
    n_hvg:Optional[int]=1000,
    bins:Optional[int]=20,
    log1p_mean_max:Optional[int]=3,
    log1p_mean_min:Optional[int]=0.0125,
    min_z_dispersion:Optional[float]=0.5,
    reindex_genes:Optional[bool]=True
    ):
    """
    Identify highly variable genes (HVGs) in single-cell RNA-seq data.
    Method used is that of Seurat [Satija, R. et al. Spatial reconstruction of single-cell gene expression data. Nat Biotechnol 33 (2015)]

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing single-cell RNA-seq data.
    n_hvg : int, optional
        Number of highly variable genes to select per sample. Defaults to 1000.
    bins : int, optional
        Number of bins to use for mean expression binning. Defaults to 20.
    log1p_mean_max : float, optional
        Maximum log1p mean expression threshold for HVG selection. Defaults to 3. Only applied if n_hvg is None.
    log1p_mean_min : float, optional
        Minimum log1p mean expression threshold for HVG selection. Defaults to 0.0125.  Only applied if n_hvg is None.
    z_dispersion_min : float, optional
        Miniumum z_dispersion. Only applied if n_hvg is None. 
    reindex_genes : bool, optional
        Whether to reindex gene indices after HVG selection. Defaults to True.

    Returns
    -------
    None
        Updates sdata.sta, sdata.X, and sdata.var with HVG selection results.
    """

    # Use gene_idx for fast groupBy operations
    sdf_stats = sdata.X.groupBy('fp_int', 'gene_idx').agg(
        F.sum('norm_counts').alias('sum_counts'),
        F.sum(F.col('norm_counts')*F.col('norm_counts')).alias('sum_squared_counts'),
        # F.variance('norm_counts').alias('var_counts')
    )
    
    # Add gene_name/id from var for later use (ordering, display)
    cols_pull_in = ['fp_int', 'gene_idx']
    if 'gene_name' in sdata.var.columns:
        cols_pull_in.append('gene_name')
    if 'gene_id' in sdata.var.columns:
        print("PULLING IN GENE ID")
        cols_pull_in.append('gene_id')
    sdf_stats = sdf_stats.join(
        sdata.var.select(*cols_pull_in),
        on=['fp_int', 'gene_idx'],
        how='left'
    )
    
    # Add file_path back to sdf_stats (lost in groupBy at line 453)
    sdf_stats = sdf_stats.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')

    # mean_counts and var_counts
    n_cells = sdata.obs.groupBy('fp_int').count().withColumnRenamed('count', 'n_cells')
    sdf_stats = sdf_stats.join(n_cells, on='fp_int', how='left')

    sdf_stats = sdf_stats.withColumn(
        'mean_counts', F.col('sum_counts') / F.col('n_cells')
    )
    sdf_stats = sdf_stats.withColumn(
        'var_counts', F.col('sum_squared_counts') / F.col('n_cells') - F.col('mean_counts')**2
    )

    # if a gene only measueres in one cell, set the variance to 0 (not null) (since mean of squares - mean**2 would be 0)
    sdf_stats = sdf_stats.withColumn('var_counts', F.when(F.col('var_counts').isNull(), 0).otherwise(F.col('var_counts')))
    
    # if any zero on mean - set to small
    sdf_stats = sdf_stats.withColumn('mean_counts', F.when(F.col('mean_counts') == 0, 1e-12).otherwise(F.col('mean_counts')))

    # Calculate dispersion
    sdf_stats = sdf_stats.withColumn('dispersion', F.col('var_counts') / F.col('mean_counts'))

    # set 0 dispersion to nan
    sdf_stats = sdf_stats.withColumn('dispersion', F.when(F.col('dispersion') == 0, F.lit(None)).otherwise(F.col('dispersion')))
    
    # Log transform dispersion and mean counts
    sdf_stats = sdf_stats.withColumn('log_dispersion', F.log(F.col('dispersion'))) # called "dispersions" in scanpy
    sdf_stats = sdf_stats.withColumn('log1p_mean', F.log1p(F.col('mean_counts'))) # called "means" in scanpy

    # Bin the genes into 20 groups based on mean counts
    # sdf_stats = sdf_stats.withColumn('mean_bin', F.ntile(20).over(Window.orderBy('log1p_mean')))

    # THIS IS NOT APPLIED PER SAMPLE CURRENTLY
    min_mean = 0.99*sdf_stats.select(F.min('log1p_mean')).collect()[0][0]
    max_mean = 1.01*sdf_stats.select(F.max('log1p_mean')).collect()[0][0]
    bin_edges = list(np.linspace(min_mean, max_mean, bins+1))
    bucketizer = Bucketizer(
        splits=bin_edges, 
        inputCol='log1p_mean', 
        outputCol='mean_bin'
    )
    sdf_stats = bucketizer.transform(sdf_stats)


    # Calculate average and standard deviation of dispersion within each bin (and file)
    sdf_bin_stats = sdf_stats.groupBy('fp_int','mean_bin').agg(
        F.mean('log_dispersion').alias('mean_dispersion'),
        F.stddev('log_dispersion').alias('std_dispersion')
    )

    # Join the bin statistics back to the original dataframe
    sdf_stats = sdf_stats.join(sdf_bin_stats, on=['fp_int', 'mean_bin'], how='left')

    # Calculate z-scored dispersion for each gene
    sdf_stats = sdf_stats.withColumn(
        'z_dispersion',
        (F.col('log_dispersion') - F.col('mean_dispersion')) / F.col('std_dispersion')
    ) # what is called "dispersions_norm" in scanpy

    
    
    # limit to n_hvg in each file
    if n_hvg is not None:
        # sdf_filtered = sdf_stats.repartition(F.col('fp_int'))
        sdf_filtered = sdf_stats.withColumn('rank', F.row_number().over(Window.partitionBy('fp_int').orderBy(F.col('z_dispersion').desc())))
        sdf_filtered = sdf_filtered.filter(F.col('rank') <= n_hvg).drop('rank')
    else:
        sdf_filtered = sdf_stats.filter(
            (sdf_stats.log1p_mean<log1p_mean_max)
            & (sdf_stats.log1p_mean>log1p_mean_min)
            & (sdf_stats.z_dispersion>min_z_dispersion)
        )

    if sdata.persist_intermediaries:
        sdf_stats.persist(sdata.persist_storage_level) # htis one may need materialize...
        sdf_filtered.persist(sdata.persist_storage_level)

    # perform filtering on sta,X,var to only leave the rows relating to these selected HGV genes
    # Use gene_idx for fast integer joins

    sdata.sta = sdf_stats.join(
        sdf_filtered.withColumn('selected',F.lit(True)).select('fp_int','gene_idx','selected'),
        on=['fp_int','gene_idx'],
        how='left'
    ).fillna({'selected': False})
    
    # print("pre-join nulls")
    # print(sdata.X.filter(F.col('gene_idx').isNull()).count())
    
    sdata.X = sdata.X.join(
        sdf_filtered.select('fp_int','gene_idx'),
        on=['fp_int','gene_idx'],
        how='inner' # testing inner again - makes more sense?
    )
    sdata.var = sdata.var.join(
        sdf_filtered.select('fp_int','gene_idx'),
        on=['fp_int','gene_idx'],
        how='inner' # testing inner again - makes more sense?
    )

    # in case any cells did not have any of the most variable genes we should remove those genes from the obs table too
    # since we will have removed them from X above

    sdata.obs = sdata.obs.join(
        sdata.X.select('fp_int','cell_idx').distinct(),
        on=['fp_int','cell_idx'],
        how='inner' # must exist in original (is cell) and in X after HVG update 
    )


    # if selected - genes will be reindexed from 0 among the remaining genes

    # print("post-join nulls")
    # print(sdata.X.filter(F.col('gene_idx').isNull()).count())
    if reindex_genes:
        # Save old gene_idx for mapping
        sdata.var = sdata.var.withColumn('old_gene_idx', F.col('gene_idx'))
        sdata.X = sdata.X.withColumn('old_gene_idx', F.col('gene_idx'))
        
        # Create new sequential gene_idx in var (ordered by gene_name for consistency)
        sdata.var = sdata.var.withColumn(
            'gene_idx',
            F.row_number().over(Window.partitionBy('fp_int').orderBy('gene_name'))
        )
        
        # Map old_gene_idx → new_gene_idx using fast integer join
        sdata.X = (
            sdata.X
            .join(
                sdata.var.select('fp_int', 'old_gene_idx', 
                               F.col('gene_idx').alias('new_gene_idx')),
                on=['fp_int', 'old_gene_idx'],
                how='left'
            )
            .drop('gene_idx', 'old_gene_idx')
            .withColumnRenamed('new_gene_idx', 'gene_idx')
        )
        
        # Clean up var
        sdata.var = sdata.var.drop('old_gene_idx')
        
        # print("post-reindex nulls")
        # print(sdata.X.filter(F.col('gene_idx').isNull()).count())
    return None




