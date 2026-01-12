from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql import Window
from typing import Optional, List
from .data import SprayData
from ._kmeans import apply_kmeans
from ._markers import get_marker_stats_pa_fn

sp_p_schema = StructType([
    StructField("fp_idx", IntegerType(), True),
    StructField("gene_idx", IntegerType(), True),
    StructField("cluster_id",IntegerType(), True),
    StructField("pvalue",DoubleType(), True),
    StructField("fc",DoubleType(), True),
])

def kmeans(sdata: SprayData, ks:Optional[List[int]]=None, seed:Optional[int]=0,max_iter:Optional[int]=20):
    """ apply kmeans (distributed) to PCA

    Will try all k in ks and return the best one (based on silhouette score) in the cluster_id column
    all k tried will be in columns f"cluster_k_{k}".

    parameters
    ----------
    sdata : SprayData
        the SprayData object
    ks : List[int]
        the list of k to try
    seed : int, optional
        the seed for kmeans, by default 0
    max_iter : int, optional
        the max iteration for kmeans, by default 20
    """
    if ks is None:
        ks = [2,3,4]

    cluster_df, scores_pdf = apply_kmeans(
        sdata=sdata,
        ks=ks,
        seed=seed,
        max_iter=max_iter
    )
    sdata.obs = sdata.obs.join(
        cluster_df.drop('file_path'),  # Drop file_path since obs already has it
        on=['fp_int','cell_idx'], 
        how='inner'
    )
    sdata.clu = cluster_df.groupBy(['fp_int', 'cluster_id']).agg(
        F.count('*').alias('n_cells')
    )
    
    # Add file_path to clu (created via groupBy)
    sdata.clu = sdata.clu.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')
    
    return scores_pdf

def rank_marker_genes(sdata:SprayData, test:str="ttest_ind",base_fdr:float = 0.05, fc_cutoff:float = 0.25, n_markers:int=10):
    """
    Ranks marker genes for each cluster in the SprayData object.

    For each cluster, selects marker genes based on statistical test, FDR, fold change, and number of markers.
    Updates sdata.clu with marker genes per cluster.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing cell and gene data.
    test : str, optional
        Statistical test to use for ranking marker genes (default: "ttest_ind", other choices = ['mannwhitneyu']).
    base_fdr : float, optional
        Base false discovery rate threshold (default: 0.05).
    fc_cutoff : float, optional
        Minimum fold change for marker selection (default: 0.25).
    n_markers : int, optional
        Number of top marker genes to select per cluster (default: 10).

    Returns
    -------
    None
    """

    marker_arrow_fn = get_marker_stats_pa_fn(test)

    sdf_inter = sdata.X.select(
        'fp_int','cell_idx','gene_idx','log1p_norm_counts'
    ).join(
        sdata.obs.select('fp_int','cell_idx','cluster_id'),
        on=['fp_int','cell_idx'],
        how='left'
    ).withColumn(
        'fp_idx',
        F.col('fp_int'),
    )

    sdf_test = sdf_inter.drop(
        'cell_idx'
    ).groupBy(
        'fp_idx','gene_idx'
    ).applyInArrow(
        marker_arrow_fn,
        schema=sp_p_schema
    ).withColumn(
        'neg_log_pvalue',
        -F.log(F.col('pvalue')+F.lit(1e-210))
    )

    window_spec = Window.partitionBy('fp_idx', 'cluster_id').orderBy(
        F.desc('fc'),
        F.desc('neg_log_pvalue')
    )

    sdf_rank = sdf_test.filter(
        (F.col('pvalue') < base_fdr/len(sdf_test.select('gene_idx').distinct().collect()))
        & (F.col('fc') > fc_cutoff)
    ).withColumn(
        'rank',
        F.row_number().over(window_spec)
    )

    # Add file_path using file_mapping (fp_idx = fp_int)
    sdf_rank = sdf_rank.join(
        F.broadcast(sdata.file_mapping.withColumnRenamed('fp_int', 'fp_idx')),
        on=['fp_idx'],
        how='left'
    )
    
    # # TODO: could do somehting like: sdata.var_clu = (?)
    
    sdf_markers = sdf_rank.filter(
        F.col('rank') <= n_markers
    ).drop('rank')

    sdf_markers = sdf_markers.join(
        sdata.var.select('fp_int','gene_idx','gene_name').withColumnRenamed('fp_int', 'fp_idx'),
        on=['fp_idx','gene_idx'],
        how='left'
    ).select(
        'fp_idx',
        'cluster_id',
        'gene_name'
    ).groupBy('fp_idx','cluster_id').agg(
        F.collect_list('gene_name').alias('marker_genes')
    )
    
    # Add file_path back (lost in groupBy at line 141)
    sdf_markers = sdf_markers.join(
        F.broadcast(sdata.file_mapping.withColumnRenamed('fp_int', 'fp_idx')), 
        on='fp_idx', 
        how='left'
    )
    
    # Rename fp_idx back to fp_int for compatibility with sdata.clu
    sdf_markers = sdf_markers.withColumnRenamed('fp_idx', 'fp_int')

    sdata.clu = sdata.clu.join(
        sdf_markers.drop('file_path'),  # Drop file_path since clu already has it
        on=['fp_int','cluster_id'],
        how='left'
    )
    return None

def as_gold_mart_data(sdata, cells_per_cluster:int=2000):
    """
    Only keep a reduced number of cells per cluster and marker genes.

    Returns a highly reduced dataset for Scanpy dashboarding.

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing cell and gene data.
    cells_per_cluster : int, optional
        Number of cells to keep per cluster (default: 2000).

    Returns
    -------
    None
    """
    sdf_clu = sdata.clu.withColumn("gene_name", F.explode("marker_genes"))

    sdata.var = sdata.var.join(
        sdf_clu.select('file_path','gene_name').dropDuplicates(),
        on=['file_path','gene_name'],
        how='inner'
    )
    
    window = Window.partitionBy('fp_int', 'cluster_id').orderBy(F.rand())
    sdata.obs = sdata.obs.withColumn(
        'row_num', F.row_number().over(window)
    ).filter(
        F.col('row_num') <= cells_per_cluster
    ).drop('row_num').cache()


    # changed order to do cells first in case lose some cells
    sdata.X = sdata.X.join(
        sdata.obs.select('fp_int','cell_idx').dropDuplicates(),
        on=['fp_int','cell_idx'],
        how='inner'
    )
    sdata.X = sdata.X.join(
        sdata.var.select('file_path','gene_name').dropDuplicates(),
        on=['file_path','gene_name'],
        how='inner'
    )
    return None

# ----- not suggested methods for scanpy soft-distributed code
# ---- prefer efficient scanpy options

from ._arrow_scanpy_clustering import apply_scanpy_clustering, apply_scanpy_pca_cluster_markers
# Backward compatibility aliases
apply_pca_cluster_markers = apply_scanpy_pca_cluster_markers
apply_clustering = apply_scanpy_clustering
