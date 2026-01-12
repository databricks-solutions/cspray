from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.functions import vector_to_array
from typing import Optional
from functools import reduce

# def pca(sdata, n_hvg:Optional[int]=None, n_components:Optional[int]=50):
    
#     sdf_per_cell = sdata.X.groupby(['file_path', 'cell_idx']).agg(
#       F.map_from_entries(
#           F.collect_list(F.struct('gene_idx', 'log1p_norm_counts'))
#         ).alias('gene_expression_map')
#     )

#     for file_path in sdata.sam.select('file_path').toPandas().file_path.unique().tolist():
        
#         if n_hvg is None:
#           fp_n_hvg = sdata.var.filter(F.col('file_path') == file_path)\
#             .select(F.max('gene_idx').alias('max_gene_idx'))\
#             .collect()[0]['max_gene_idx']+1
#         #   print(file_path, fp_n_hvg)
#         else:
#           fp_n_hvg = n_hvg
        
#         sparse_vectors = (
#             sdf_per_cell.filter(F.col('file_path') == file_path)
#             .withColumn(
#               'sparse_vector', 
#               F.udf(lambda gene_expression_map: SparseVector(fp_n_hvg,gene_expression_map),
#                     VectorUDT())('gene_expression_map')
#             )
#         )
        
#         # Perform PCA on the sparse vectors
#         pca_model = PCA(k=50, inputCol='sparse_vector', outputCol='pca_features')
#         pca_df = pca_model.fit(sparse_vectors).transform(sparse_vectors)
        
#         # append pca_df into a spark df to collect from each file_path
#         if 'final_pca_df' not in locals():
#             final_pca_df = pca_df
#         else:
#             final_pca_df = final_pca_df.union(pca_df)
    
#     final_pca_df = final_pca_df.withColumn(   
#       'pca_features_array', 
#       vector_to_array("pca_features")
#     ).drop( 'sparse_vector', 'pca_features', 'gene_expression_map')
#     return final_pca_df
  
def pca(sdata, n_hvg:Optional[int]=1000, n_components:Optional[int]=50, hard_repartion_n:Optional[int]=None):
    """
    Perform PCA on per-cell gene expression data from multiple files.

    Parameters
    ----------
    sdata : object
        An object containing Spark DataFrames 'X' (expression data), 'sam' (sample metadata), and 'var' (gene metadata).
    n_hvg : int, optional
        Number of highly variable genes to use. Defaults to 1000.
    n_components : int, optional
        Number of PCA components. Defaults to 50.
    hard_repartion_n : int, optional
        If set, repartitions data per file before PCA for parallelism. Not required if Delta table is stored post-HVG step.

    Returns
    -------
    DataFrame
        Spark DataFrame containing per-cell PCA features as arrays, with metadata columns.
    """
    
    sdf_per_cell = sdata.X.groupby(['fp_int', 'cell_idx']).agg(
      F.map_from_entries(
          F.collect_list(F.struct('gene_idx', 'log1p_norm_counts'))
        ).alias('gene_expression_map')
    )
    
    # Add file_path back (needed for filter at line 98)
    sdf_per_cell = sdf_per_cell.join(F.broadcast(sdata.file_mapping), on='fp_int', how='left')
    
    # partition_lengths = sdf_per_cell.rdd.glom().map(len).collect()
    # print("partitions for pre vector")
    # print(partition_lengths)
    sparse_vectors_all = (
        sdf_per_cell\
        .withColumn(
          'sparse_vector', 
          F.udf(lambda gene_expression_map: SparseVector(n_hvg+1,gene_expression_map),
                VectorUDT())('gene_expression_map')
        )
    )
    # partition_lengths = sparse_vectors_all.rdd.glom().map(len).collect()
    # print("partitions for all")
    # print(partition_lengths)
      
    
    pca_dfs = []
    files = sdata.sam.select('file_path').toPandas().file_path.unique().tolist()
    for file_path in files:
        print("doing : ", file_path)
        sparse_vectors = sparse_vectors_all.filter(F.col('file_path') == file_path)
        if hard_repartion_n:
            sparse_vectors = sparse_vectors.repartition(hard_repartion_n)
        
        # partition_lengths = sparse_vectors.rdd.glom().map(len).collect()
        # print(partition_lengths)

        # Perform PCA on the sparse vectors
        pca_model = PCA(k=50, inputCol='sparse_vector', outputCol='pca_features')
        pca_df = pca_model.fit(sparse_vectors).transform(sparse_vectors)
        pca_dfs.append(pca_df)
        # # append pca_df into a spark df to collect from each file_path
        # if 'final_pca_df' not in locals():
        #     final_pca_df = pca_df
        # else:
        #     final_pca_df = final_pca_df.union(pca_df)
    
    final_pca_df = reduce(
        lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), 
        pca_dfs
    )
    final_pca_df = final_pca_df.withColumn(   
      'pca_features_array', 
      vector_to_array("pca_features")
    ).drop('sparse_vector', 'pca_features', 'gene_expression_map', 'file_path')
    return final_pca_df
  


