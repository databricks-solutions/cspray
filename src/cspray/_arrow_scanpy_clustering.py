import anndata as ad
import pandas as pd
import pyarrow as pa
import numpy as np
from pyspark.sql.types import *
from pyspark.sql import functions as F
from typing import Optional
import scanpy
from .data import SprayData

pa_schema = pa.schema([
    pa.field('fp_idx', pa.int32()),
    pa.field('cell_idx', pa.int64()),
    pa.field('PCs', pa.list_(pa.float32())),
    pa.field('cluster_id', pa.int32()),
    pa.field('marker_genes', pa.list_(pa.string()))
])
# same schema but pyspark
sp_schema = StructType([
    StructField("fp_idx", IntegerType(), True),
    StructField("cell_idx", LongType(), True),
    StructField("PCs", ArrayType(FloatType()), True),
    StructField("cluster_id",IntegerType(), True),
    StructField("marker_genes", ArrayType(StringType()), True),
])

pa_cl_schema = pa.schema([
    pa.field('fp_idx', pa.int32()),
    pa.field('cell_idx', pa.int64()),
    pa.field('cluster_id', pa.int32())
])
# same schema but pyspark
sp_cl_schema = StructType([
    StructField("fp_idx", IntegerType(), True),
    StructField("cell_idx", LongType(), True),
    StructField("cluster_id",IntegerType(), True),
])



def get_pure_clustering_arrow_fn(
    pca_n_comps:Optional[int] = 50,
    cluster_resolution:Optional[float] = 0.1,
    ):
    """
    Processes an Arrow Table to assign clusters using PCA features and returns a new Arrow Table with cluster assignments.

    Used to get a function that will process arrow frame to add clustering info
     - will use scanpy's functionailty for this
     - PCA information will already be in the arrow frame as an array
       - required columns in arrow table:
       - fp_idx is the file_path index and there willl only be one as this fiunction should be applied folllowing a groupby
       - cell_idx is the cell index for each cell (for later join)
       - pca_features_array

    NOTE: this requires a seperate igraph install should you wish to use this 
    
    Parameters
    ----------
    pca_n_comps : int, optional
        Number of principal components to use (default: 50).
    cluster_resolution : float, optional
        Resolution parameter for Leiden clustering (default: 0.1).

    Returns
    -------
    arrow_to_clusters : Callable
        Function that takes a pyarrow.Table and returns a pyarrow.Table with columns ['fp_idx', 'cell_idx', 'cluster_id'].
    """
    
    def arrow_to_clusters(table: pa.Table) -> pa.Table:
        sample_idx = table.column("fp_idx")[0].as_py()
        # table = table.select(['cell_idx','pca_feature_array'])
        
        # df = table.to_pandas()
        # pca_arr = np.vstack(df['pca_features_array'].to_numpy())
        # adata = ad.AnnData(
        #     obs=pd.DataFrame(
        #         index=df.cell_idx,
        #         columns=['batch'], 
        #         data={'batch':[0]*len(df)}
        #     )
        # )

        pca_arr = np.vstack(table.column('pca_features_array').to_numpy())
        cell_idx = table.column('cell_idx').to_numpy()
        adata = ad.AnnData(
            obs=pd.DataFrame(
                index=cell_idx,
                columns=['batch'], 
                data={'batch':[0]*len(cell_idx)}
            )
        )

        adata.obsm = {'X_pca': pca_arr}
       
        
        # or bbknn? may be faster
        # scanpy.pp.neighbors(adata, use_rep='X_pca')
        scanpy.external.pp.bbknn(adata,batch_key='batch',computation="annoy", use_annoy=None, approx=None, use_faiss=None) #use_annoy=False)

        scanpy.tl.leiden(
            adata,
            resolution=cluster_resolution,
            random_state=0,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        adata.obs['fp_idx'] = sample_idx
        
        # pull out obs and ensure get cell_idx out
        df_out = adata.obs.reset_index()\
            .rename(columns={"index":"cell_idx"})
        
        # force int types
        df_out['cell_idx'] = df_out['cell_idx'].astype(int)
        df_out['leiden'] = df_out['leiden'].astype(int)

        # just in case anndata ever add addditional cols, keep only our schema
        df_out = df_out[[
            'fp_idx',
            'cell_idx',
            'leiden',
        ]].rename(columns={'leiden':'cluster_id'})

        # df_out = pd.DataFrame([{
        #     'fp_idx': sample_idx,
        #     'cell_idx': 0,
        #     'cluster_id': 0,
        # }])

        table = pa.Table.from_pandas(df_out, schema=pa_cl_schema)
        return table
    
    return arrow_to_clusters

def apply_scanpy_clustering(
    sdata : SprayData,
    pca_n_comps:Optional[int] = 50,
    cluster_resolution:Optional[float] = 0.1,  # low res on purpose (larger clusters)
    cache_intermediary:Optional[bool] = True,
    ):
    """
    This clustering function performs clustering using scanpy cluserting method (Leiden) under assumption PCA is precalculated

     - note that because groupby over samples could include large samples that even after HVG restriction may still cause OOM errors
     - should you try this and get OOM prefer the PCA and KMeans clustering methods provided in the main example of this repo
    
    NOTE: this requires a seperate igraph install should you wish to use this 

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing cell and gene data.
    pca_n_comps : int, optional
        Number of principal components to use (default: 50).
    cluster_resolution : float, optional
        Resolution parameter for Leiden clustering (default: 0.1).
    cache_intermediary : bool, optional
        Whether to cache intermediary Spark DataFrames (default: True).

    Returns
    -------
    None
    """
    fn = get_pure_clustering_arrow_fn(
        pca_n_comps=pca_n_comps,
        cluster_resolution=cluster_resolution,
    )

    sdf_inter = sdata.obs.select(
        'file_path',
        'cell_idx',
        'pca_features_array',
    ).withColumn(
        'fp_idx',
        F.hash(F.col('file_path')),
    )

    sdf_proc = sdf_inter.drop('file_path').groupby('fp_idx').applyInArrow(
        fn,
        schema=sp_cl_schema
    )
    
    if cache_intermediary:
        sdf_proc = sdf_proc.cache()

    sdf_proc = sdf_proc.join(
        sdf_inter.select('file_path','fp_idx'),
        on='fp_idx',
        how='left'
    ).drop('fp_idx')

    # map PCs and cluster idxs into cell df in sdata
    sdata.obs = sdata.obs.join(
        sdf_proc.select('file_path','cell_idx','cluster_id'),
        on=['fp_int','cell_idx']
    )

    # map counts and markers into cluster df in sdata
    sdata.clu = sdf_proc.groupBy(['fp_int', 'cluster_id']).agg(
         F.count('*').alias('count')
    )
    
    # Add file_path to clu (created via groupBy)
    sdata.clu = sdata.clu.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')
    
    return None

def get_clustering_arrow_fn(
    pca_n_comps:Optional[int] = 50,
    cluster_resolution:Optional[float] = 0.1,
    n_markers:Optional[int]=5,
    markers_test: Optional[str] = "wilcoxon",
    ):
    """ 
    This clustering arrow function collector is for PCA/clustering/markers as single shot
     - note that because groupby over samples could include large samples that even after HVG restriction may still cause OOM errors
     - should you try this and get OOM prefer the PCA and KMeans clustering methods provided in the main example of this repo.
    
    NOTE: this requires a seperate igraph install should you wish to use this 

    Parameters
    ----------
    pca_n_comps : int, optional
        Number of principal components to use (default: 50).
    cluster_resolution : float, optional
        Resolution parameter for Leiden clustering (default: 0.1).
    n_markers : int, optional
        Number of top marker genes to select per cluster (default: 5).
    markers_test : str, optional
        Statistical test to use for ranking marker genes (default: "wilcoxon").

    Returns
    -------
    arrow_hvg_to_cluters : Callable
        Function that takes a pyarrow.Table and returns a pyarrow.Table with columns ['fp_idx', 'cell_idx', 'PCs', 'leiden', 'marker_genes'].
    
    """
    def arrow_hvg_to_cluters(table: pa.Table) -> pa.Table:
        
        # df = table.to_pandas()
        # sample_idx = df['file_path'].iloc[0]
        
        sample_idx = table.column("fp_idx")[0].as_py()
        table = table.select(['cell_idx','gene_name','log1p_norm_counts'])
        

        # # -- pandas way -----
        table = table.to_pandas()
        table = table.pivot(index='cell_idx', columns='gene_name', values='log1p_norm_counts')
        table = table.fillna(0.0) #0 into log(x+1) is still 0 (NaN are o's from sparse)
        
        # Create an AnnData object
        adata = ad.AnnData(table)
        adata.obs = pd.DataFrame({'cell_idx': table.index})
        adata.var = pd.DataFrame({'gene_name': table.columns})
        adata.var.index = table.columns
        
        # -- sparse array way -----
        # dictionary_array = table.column('gene_name').combine_chunks().dictionary_encode()
        # table = table.append_column('gene_idx', dictionary_array.indices)

        # n_cells = pa.compute.max(table.column('cell_idx')).as_py() + 1
        # n_genes = pa.compute.max(table.column('gene_idx')).as_py() + 1

        # sparse_matrix = csr_matrix((table.column('log1p_norm_counts'), (table.column('cell_idx'), table.column('gene_idx'))), shape=(n_cells, n_genes))

        # # obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_cells)])
        # obs = table.select(['cell_idx']).to_pandas()\
        #     .drop_duplicates(subset='cell_idx', keep='first')\
        #     .sort_values(by='cell_idx')

        # var = table.select(['gene_idx','gene_name']).to_pandas()\
        #     .drop_duplicates(subset='gene_idx', keep='first')\
        #     .sort_values(by='gene_idx')

        # # not good idea in a UDF?
        # # del table
        # # gc.collect()

        # adata = ad.AnnData(X=sparse_matrix, obs=obs, var=var)

        # # artificial limit for testing
        # # adata = adata[:100,:]

        # -------continue ------------

        scanpy.tl.pca(adata, n_comps=pca_n_comps, svd_solver="arpack")
        scanpy.pp.neighbors(adata)

        scanpy.tl.leiden(
            adata,
            resolution=cluster_resolution, # low res on purpose (larger clusters)
            random_state=0,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        scanpy.tl.rank_genes_groups(adata, "leiden", method=markers_test) # t test is faster
        
        # top 5 markers for each clutser (cluster are columns - we don't know the number of clusters)
        marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(n_markers).T
        # merge PCA and cluster information into adata.obs
        adata.obs['PCs'] = pd.DataFrame({'PCs': [row for row in adata.obsm['X_pca']]}, index=adata.obs.index)

        adata.obs['leiden'] = adata.obs['leiden'].astype(int)
        # now for marker gene table, make each row an array of the rows, and then join in on index to leiden
        marker_df = marker_df.apply(lambda x: x.tolist(), axis=1).to_frame()
        
        # ensure column name is marker_genes
        marker_df.columns = ['marker_genes']
        marker_df.index.name = 'leiden'
        marker_df = marker_df.reset_index() 
        marker_df['leiden'] = marker_df['leiden'].astype(int)
        
        adata.obs = adata.obs.merge(
            marker_df, 
            on='leiden', 
            how='left'
        )
        # convert to pyarrow table, set schema too
        adata.obs['fp_idx'] = sample_idx
        df_out = adata.obs.reset_index()
        df_out = df_out[[
            'fp_idx',
            'cell_idx',
            'PCs',
            'leiden',
            'marker_genes'
        ]].rename(columns = {'leiden':'cluster_id'})

        table = pa.Table.from_pandas(df_out, schema=pa_schema)
        return table
    
    return arrow_hvg_to_cluters

def apply_scanpy_pca_cluster_markers(
    sdata : SprayData,
    pca_n_comps:Optional[int] = 50,
    cluster_resolution:Optional[float] = 0.1,
    n_markers:Optional[int]=10,
    markers_test: Optional[str] = "wilcoxon",
    cache_intermediary:Optional[bool] = True,
    ):
    """

    Performs PCA, clustering, and marker gene identification in a single step.

    This clustering method is for PCA/clustering/markers as single shot
     - note that because groupby over samples could include large samples that even after HVG restriction may still cause OOM errors
     - should you try this and get OOM prefer the PCA and KMeans clustering methods provided in the main example of this repo

    NOTE: this requires a seperate igraph install should you wish to use this     

    Parameters
    ----------
    sdata : SprayData
        The SprayData object containing cell and gene data.
    pca_n_comps : int, optional
        Number of principal components to use (default: 50).
    cluster_resolution : float, optional
        Resolution parameter for Leiden clustering (default: 0.1).
    n_markers : int, optional
        Number of top marker genes to select per cluster (default: 10).
    markers_test : str, optional
        Statistical test to use for ranking marker genes (default: "wilcoxon").
    cache_intermediary : bool, optional
        Whether to cache intermediary Spark DataFrames (default: True).

    Returns
    -------
    None
    """

    fn = get_clustering_arrow_fn(
        pca_n_comps=pca_n_comps,
        cluster_resolution=cluster_resolution,
        n_markers=n_markers,
        markers_test=markers_test
    )

    sdf_inter = sdata.X.select(
        'file_path',
        'cell_idx',
        'log1p_norm_counts',
        'gene_name'
    ).withColumn(
        'fp_idx',
        F.hash(F.col('file_path')),
    )

    sdf_proc = sdf_inter.drop('file_path').groupby('fp_idx').applyInArrow(
        fn,
        schema=sp_schema
    )
    
    if cache_intermediary:
        sdf_proc = sdf_proc.cache()

    sdf_proc = sdf_proc.join(
        sdf_inter.select('file_path','fp_idx'),
        on='fp_idx',
        how='left'
    ).drop('fp_idx')

    # map PCs and cluster idxs into cell df in sdata
    sdata.obs = sdata.obs.join(
        sdf_proc.select('file_path','cell_idx','PCs','cluster_id'),
        on=['fp_int','cell_idx']
    )

    # map counts and markers into cluster df in sdata
    sdata.clu = sdf_proc.groupBy(['fp_int', 'cluster_id']).agg(
        F.first('marker_genes').alias('marker_genes')
    )
    
    # Add file_path to clu (created via groupBy)
    sdata.clu = sdata.clu.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')
    
    return None
    