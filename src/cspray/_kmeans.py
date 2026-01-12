from .data import SprayData
from pyspark.ml.functions import array_to_vector
from pyspark.ml.clustering import KMeans
from functools import reduce
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np
from pyspark.sql import functions as F
from typing import Optional,List
import pandas as pd

def apply_kmeans(sdata: SprayData, ks:List[int], seed:int=0,max_iter:int=20):
    """
    Applies KMeans clustering to PCA features for each distinct file_path in SprayData.

    Parameters
    ----------
    sdata : SprayData
        SprayData object of the scRNA dataset, expects PCA to have been performed
    ks : List[int]
        List of cluster counts to evaluate with KMeans.
    seed : int, optional
        Random seed for KMeans initialization (default is 0).
    max_iter : int, optional
        Maximum number of iterations for KMeans (default is 20).

    Returns
    -------
    cluster_df : pyspark.sql.DataFrame
        DataFrame with cluster assignments for each cell, including the best cluster_id per file_path.
    scores_pdf : pandas.DataFrame
        DataFrame with silhouette scores for each file_path and k.
    """
    file_paths = sdata.obs.select('file_path').distinct().toPandas()['file_path'].values

    sdfs = []
    scores = dict()
    for fp in file_paths:
        scores[fp] = []
        internal_dfs = []
        sdf = sdata.obs.filter(sdata.obs.file_path == fp)\
                .withColumn('features', array_to_vector('pca_features_array'))\
                .select('fp_int','file_path','cell_idx','features')
        for k in ks:
            kmeans = KMeans(
                k=k,
                seed=seed,
                maxIter=max_iter,
                featuresCol='features', 
                predictionCol=f'cluster_k_{k}',
            )
            sdf_fit = kmeans.fit(sdf).transform(sdf) 
            evaluator = ClusteringEvaluator(metricName="silhouette", predictionCol=f'cluster_k_{k}')
            score = evaluator.evaluate(sdf_fit)
            scores[fp].append(score)
            internal_dfs.append(sdf_fit.drop('features', 'file_path'))
        
        fp_df = reduce(
            lambda df1, df2: df1.join(
                df2, 
                on= ['fp_int','cell_idx'],
                how= 'inner'
            ), 
            internal_dfs
        )
        
        # Add file_path back to fp_df (all rows have same file_path = fp)
        fp_df = fp_df.withColumn('file_path', F.lit(fp))
        
        best_k = ks[np.argmax(scores[fp])]
        print(best_k)
        print(scores[fp])
        fp_df = fp_df.withColumn(
            'cluster_id', 
            F.col(f"cluster_k_{best_k}")
        )

        sdfs.append(fp_df)
                
    cluster_df = reduce(
        lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), 
        sdfs
    )
    
    if sdata.persist_intermediaries:
        cluster_df.persist(sdata.persist_storage_level)

    score_tuples=[(fp, k, score) for fp, scores_values in scores.items() for k, score in zip(ks,scores_values)]
    scores_pdf = pd.DataFrame(score_tuples, columns=['file_path','k','score'])
    # scores_pdf = (
    #     spark.createDataFrame({
    #         [(fp, k, score) for fp, scores_values in scores.items() for k, score in zip(ks,scores_values) ],
    #         ["file_path", "k", "score"]
    #     )
    # )
    return cluster_df, scores_pdf





