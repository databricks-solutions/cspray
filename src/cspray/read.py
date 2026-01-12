from pyspark.sql import functions as F
from pyspark.sql import types as T
import pyspark
from typing import Iterator,Tuple,Optional,Union,List
import h5py
import numpy as np
from scipy import sparse
import pandas as pd
import copy
import pyarrow as pa
import logging
import os
import warnings

from .utils import h5ad_format_check

expstruct_wfile_intfn = T.StructType([
    T.StructField("fp_int", T.IntegerType(), False),
    T.StructField("row_idx", T.LongType(), False),
    T.StructField("col_idx", T.LongType(), False),
    T.StructField("expression", T.FloatType(), False),
])

pa_int_schema = pa.schema([
    ('fp_int',pa.int32()),
    ('row_idx',pa.int64()),
    ('col_idx',pa.int64()),
    ('expression',pa.float32())
])

pa_gene_schema = pa.schema([
    ('fp_int',pa.int32()),
    ('gene_idx',pa.int64()),
    ('gene_id',pa.string()),
    ('gene_name',pa.string())
])
genestruct = T.StructType([
    T.StructField("fp_int", T.IntegerType(), False),
    T.StructField("gene_idx", T.LongType(), False),
    T.StructField("gene_id", T.StringType(), False),
    T.StructField("gene_name", T.StringType(), False),
])

pa_cell_schema = pa.schema([
    ('fp_int',pa.int32()),
    ('cell_idx',pa.int64()),
    ('cell_barcode',pa.string()),
])
cellstruct = T.StructType([
    T.StructField("fp_int", T.IntegerType(), False),
    T.StructField("cell_idx", T.LongType(), False),
    T.StructField("cell_barcode", T.StringType(), False),
])



def default_read_fn(spark, name: str):
    return spark.table(name)

def path_read_fn(spark, name: str):
    return spark.read.format("delta").load(name)

DEFAULT_READERS = {
    'databricks':default_read_fn,
    'delta':path_read_fn
}



@F.udf(returnType=T.LongType())
def udf_get_default_maxsize(path):
    file = h5py.File(path, 'r') 
    try:
        size = file['X']['data'].shape[0]
    except:
        warnings.warn('default read failing (ie from X in h5ad) - assuming raw was used and thats why default is failing and you`ll be ok bc you`re using raw')
        # X doesn't exist or is some empty structure - this shoudl not happen unless data in raw anyway
        return 0
    return size

@F.udf(returnType=T.LongType())
def udf_get_raw_maxsize(path):
    file = h5py.File(path, 'r') 
    if 'raw' not in file.keys():
        return 0
    size = file['raw']['X']['data'].shape[0]
    return size

def coo_subarr_to_arrmap(coo_chunk):
    arr_out = [{
        'row_idx': int(coo_chunk.row[i]),
        'col_idx':int(coo_chunk.col[i]),
        'expression':float(coo_chunk.data[i])} 
        for i in range(coo_chunk.nnz)]
    return arr_out

def coo_subarr_to_arrmap_int(coo_chunk):
    arr_out = [{
        'row_idx': int(coo_chunk.row[i]),
        'col_idx':int(coo_chunk.col[i]),
        'expression':int(coo_chunk.data[i])} 
        for i in range(coo_chunk.nnz)]
    return arr_out

def get_csr_submatrix_from_raw(
    data:np.ndarray,
    indices:np.ndarray,
    indptr:np.ndarray,
    start_idx:int,
    end_idx:int,
    backed:Optional[bool]=True,
    tocoo:Optional[bool]=True,
    ):
    """ get a submatrix from a csr matrix
    parameters
    -------
    data: array of values, can be backed on disk (e.g h5file)
    indices: array of indices, can be backed on disk (e.g h5file)
    indptr: array of indices, can be backed on disk (e.g h5file)
    start_idx: int, start index of submatrix (within data)
    end_idx: int, end index of submatrix (within data)
    backed: bool, whether to use a backed array
    tocoo: bool, whether to return a coo matrix

    notes
    -----

    start and end indices are inclusive
    could parallelize over threads 
    """
    
    if backed:
        indptr = indptr[:] # need entire row for np operations (also is smallest data piece )

    # first row_pointer less than start_idx
    rowstart = np.argmin(indptr<=start_idx)-1

    # one prior to the first value greater than end_idx
    # this can be apointer to value equal to end_idx
    rowend = np.argmax(indptr>end_idx)-1
    if rowend==-1:
        # means did not find end_idx (ie no number that big)
        # WARN...?
        rowend = len(indptr)-2 # largest row index (recall last is just nnz number)
        end_idx = indptr[-1]-1

    # make a modifiable copy of row pointers
    # must correct for submatrix
    myrows = copy.deepcopy(indptr[rowstart:rowend+1])
    myrows[0] = start_idx
    myrows -= start_idx
    myrows = np.concatenate((myrows, np.array([end_idx-start_idx+1])))
    
    mm = sparse.csr_matrix( (
        data[start_idx:end_idx+1],
        indices[start_idx:end_idx+1],
        myrows 
        ) 
    )
    if tocoo:
        mm = mm.tocoo()
        mm = sparse.coo_array((
            mm.data,
            (mm.row+rowstart,mm.col)
        ))
    return mm

def mapinarrow_process_float_expression_h5ad(itr: Iterator, from_raw:bool=True, fallback_default:bool=False):
    """
    Processes batches of file paths and indices to extract expression data from h5ad files as Arrow RecordBatches.

    Parameters
    ----------
    itr : Iterator
        Iterator over batches containing file paths and index ranges.
    from_raw : bool, optional
        If True, attempts to read from the 'raw' group in the h5ad file.
    fallback_default : bool, optional
        If True and 'raw' group is missing, falls back to the default group.

    Yields
    ------
    pa.RecordBatch
        Arrow RecordBatch containing fp_int, row_idx, col_idx, and expression for each entry in the chunk.
    """
    for batch in itr:
        d = batch.to_pydict()
        for file_path,fp_int,start_idx,end_idx in zip(d['file_path'], d['fp_int'], d['start_idx'], d['end_idx']):
            file = h5py.File(file_path, 'r')
            coo_chunk = None
            if from_raw:       
                if 'raw' in file.keys():
                    coo_chunk = get_csr_submatrix_from_raw(
                        file['raw']['X']['data'],
                        file['raw']['X']['indices'],
                        file['raw']['X']['indptr'],
                        start_idx,
                        end_idx,
                        backed=True,
                        tocoo=True
                    ) 
                else: 
                    if fallback_default:
                        coo_chunk = get_csr_submatrix_from_raw(
                            file['X']['data'],
                            file['X']['indices'],
                            file['X']['indptr'],
                            start_idx,
                            end_idx,
                            backed=True,
                            tocoo=True
                        )  
            else:
                coo_chunk = get_csr_submatrix_from_raw(
                    file['X']['data'],
                    file['X']['indices'],
                    file['X']['indptr'],
                    start_idx,
                    end_idx,
                    backed=True,
                    tocoo=True
                )  
            yield pa.RecordBatch.from_pydict({
                'fp_int': [fp_int]*coo_chunk.nnz,
                'row_idx': coo_chunk.row,
                'col_idx': coo_chunk.col,
                'expression': coo_chunk.data
            }, schema=pa_int_schema)

def process_type(v, limit : int =None):
    """
    Processes an H5AD dataset or array-like object, returning its contents as a NumPy array or string array.

    Parameters
    ----------
    v : h5py.Dataset or array-like
        The dataset or array to process.
    limit : int, optional
        If provided, limits the number of elements returned.

    Returns
    -------
    np.ndarray or np.chararray
        The processed array, either as a string array (if dtype is 'S' or 'O') or as a regular NumPy array.
    """
    if limit is None:
        if v.dtype.kind in ['S','O']:
            return v.asstr()[:]   
        else:
            return v[:]
    else:
        if v.dtype.kind in ['S','O']:
            return v.asstr()[:limit]   
        else:
            return v[:limit]
    
def h5_group_pdf_to_dict(group, limit : int = None, keys : Optional[List[str]] = None):
    """
    Converts an H5AD group to a dictionary of arrays or Series. Useful for dataframe type extraction.

    Parameters
    ----------
    group : h5py.Group
        The HDF5(h5ad) group to convert.
    limit : int, optional
        If provided, limits the number of elements returned for each key.

    Returns
    -------
    dict
        Dictionary mapping keys to arrays or pandas Series.
    """
    data_dict = {}

    if keys is None:
        keys = list(group.keys())

    for key in group.keys():
        try:
            data_dict[key] = process_type(group[key], limit=limit)
        except:
            try:
                data_dict[key] = pd.Series(pd.Categorical.from_codes(
                    process_type(group[key]['codes']),
                    process_type(group[key]['categories'])
                ))
                if limit is not None:

                    data_dict[key] = data_dict[key].head(limit)
            except Exception as e:
                logging.warn(f"key: {key}, does not behave according to expected rules. Error: {e}")
    return data_dict


def mapinarrow_var_from_h5ad(itr: Iterator, gene_name_column:Optional[str]=None, from_raw:bool=True, fallback_default:bool=False):
    """
    Extracts gene metadata from h5ad files as Arrow RecordBatches.

    Parameters
    ----------
    itr : Iterator
        Iterator over batches containing file paths and file integer identifiers.
    gene_name_column : str, optional
        Name of the column to use for gene names. If None, only gene IDs are extracted.
    from_raw : bool, optional
        If True, attempts to read from the 'raw' group in the h5ad file.
    fallback_default : bool, optional
        If True and 'raw' group is missing, falls back to the default group.

    Yields
    ------
    pa.RecordBatch
        Arrow RecordBatch containing fp_int, gene_idx, gene_id, and gene_name for each gene in the file.
    """
    # def mapinarrow_var_from_h5ad(itr: Iterator):
    for batch in itr:
        d = batch.to_pydict()
        for file_path,fp_int in zip(d['file_path'], d['fp_int']):
            file = h5py.File(file_path, 'r')
            
            # parse var from h5 to dict
            tmp_dict = None
            if from_raw:       
                if 'raw' in file:
                    tmp_dict = h5_group_pdf_to_dict(file['raw']['var']) 
                    col_map = {file['raw']['var'].attrs['_index']:'gene_index'}
                else:
                    if fallback_default:
                        tmp_dict = h5_group_pdf_to_dict(file['var']) 
                        col_map = {file['var'].attrs['_index']:'gene_index'}
                    else:
                        raise Exception("No raw data found in h5ad file. Please set fallback_default=True to use the default data instead. Or ensure your h5ad files all have raw if using raw")
            else:
                    tmp_dict = h5_group_pdf_to_dict(file['var'])
                    col_map = {file['var'].attrs['_index']:'gene_index'}
            
            # convert to df
            tmp_df = pd.DataFrame(tmp_dict)
            tmp_df = tmp_df.rename(columns=col_map)

            # identify the Ensembl_id column
            # col_map = None
            # for c in tmp_df.columns:
            #     if tmp_df[c].head(200).astype(str).str.startswith('ENSG').mean()>0.2:
            #         if c != 'gene_index':
            #             col_map = {c:'gene_index'}
            # if col_map is not None:
            #     tmp_df = tmp_df.rename(columns=col_map)
            
            
            # extract only ensembl_ids and gene_names (really only need the ensembl as later will use together with a consistent mapping)
            if gene_name_column is not None:
                tmp_df = tmp_df[['gene_index',gene_name_column]]
            else:
                tmp_df = tmp_df[['gene_index']]
            
            ensembl_ids = tmp_df['gene_index'].values
            
            if gene_name_column is not None:
                gene_names = tmp_df[gene_name_column].values
            else:
                gene_names = tmp_df['gene_index'].values
            logging.info(tmp_df.head())
            # ensemble_ids = np.char.decode(file['raw']['var']['feature_id'][:].astype('S'))
            # gene_names = np.char.decode(file['raw']['var']['feature_names'][:].astype('S'))
            yield pa.RecordBatch.from_pydict({
                'fp_int': [fp_int]*len(ensembl_ids),
                'gene_idx' : np.arange(len(ensembl_ids)),
                'gene_id': ensembl_ids,
                'gene_name': gene_names,
            }, schema=pa_gene_schema)
        # return internal_var_from_h5ad

def construct_h5ad_path_df(
    path:Optional[Union[List,str]]=None, 
    df:Optional[pyspark.sql.DataFrame]=None, 
    spark:Optional[pyspark.sql.session.SparkSession]=None,
    ):
    """
    Constructs a Spark DataFrame of h5ad file paths for downstream processing.

    Parameters
    ----------
    path : str or list of str, optional
        Directory containing h5ad files, a single h5ad file path (could be globbed with *), or a list of h5ad file paths.
    df : pyspark.sql.DataFrame, optional
        Existing DataFrame containing file paths. If provided, 'path' must be None.
    spark : SparkSession, optional
        Spark session used to create DataFrame from file paths.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with a 'file_path' column containing h5ad file paths.
    """
    if path is not None:
        if df is None:
            if isinstance(path, str):
                if not path.endswith('.h5ad'): # assume directory
                    df = pd.DataFrame({'file_path': [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5ad')]})
                else: # sigle h5ad file
                    df = pd.DataFrame({'file_path': [path]})
            elif isinstance(path, list):
                df = pd.DataFrame({'file_path': path})
            else:
                raise ValueError("path must be a string or a list of strings")
            df = spark.createDataFrame(df)
        else:
            raise ValueError("provide only path or df, not both")
    return df

def read_expression_from_h5ads(
    spark:pyspark.sql.session.SparkSession,
    path:Optional[Union[List,str]]=None, 
    df:Optional[pyspark.sql.DataFrame]=None, 
    force_partitioning: Optional[int]=None,
    chunk_size: Optional[int]=30_000_000,
    from_raw:bool=True,
    fallback_default:bool=False
    ):
    """
    Reads expression data from one or more h5ad files into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session to use for DataFrame operations.
    path : str or list of str, optional
        Directory containing h5ad files, a single h5ad file path, or a list of h5ad file paths.
    df : pyspark.sql.DataFrame, optional
        Existing DataFrame containing file paths. If provided, 'path' must be None.
    force_partitioning : int, optional
        If provided, repartitions the DataFrame to the specified number of partitions.
    chunk_size : int, optional
        Number of expression entries to read per chunk from each file.
    from_raw : bool, optional
        If True, reads expression data from the 'raw' group in the h5ad file.
    fallback_default : bool, optional
        If True and 'raw' group is missing, falls back to the default group.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame containing expression data with columns: file_path, fp_int, cell_idx, gene_idx, expression.
    """
    df = construct_h5ad_path_df(path,df,spark)
    h5ad_format_check(df, from_raw, fallback_default)
    
    df = df.select('file_path').distinct()
    df = df.withColumn('fp_int', F.hash('file_path'))

    # get the file size 
    # - use raw group if requested or eslse default groups
    # - use default group if fallback_default is True and from_raw is true
    # suggest only using raw or not raw for all files and doing upstream processing as required
    if from_raw:
        sdf = df.withColumn('raw_maxsize', udf_get_raw_maxsize(F.col('file_path')))
        if fallback_default:
            sdf = sdf.withColumn('default_maxsize', udf_get_default_maxsize(F.col('file_path')))
            sdf = sdf.withColumn('maxsize', F.when(F.col('raw_maxsize') == 0, F.col('default_maxsize')).otherwise(F.col('raw_maxsize')))
        else:
            sdf = sdf.withColumn('maxsize', F.col('raw_maxsize'))
    else:
        sdf = df.withColumn('maxsize', udf_get_default_maxsize(F.col('file_path')))

    sdf = sdf.withColumn('indices', F.sequence(F.lit(0), F.col('maxsize'), F.lit(chunk_size)))
    sdf = sdf.withColumn('start_idx', F.explode('indices'))
    if force_partitioning:
        sdf = sdf.withColumn("id", F.monotonically_increasing_id()).repartition(force_partitioning)
    sdf = sdf.withColumn(
        'end_idx', 
        F.least(F.col('maxsize'),F.col('start_idx')+F.lit(chunk_size-1))
    ).drop(
        'indices'
    )
    sdf = sdf.select('file_path','fp_int','start_idx','end_idx')\
        .mapInArrow(
            lambda x: mapinarrow_process_float_expression_h5ad(x, from_raw=from_raw, fallback_default=fallback_default),
            schema=expstruct_wfile_intfn
        )

    sdf = sdf.join(
        df,
        how='left',
        on='fp_int'
    )
    sdf = sdf.withColumnRenamed('col_idx', 'gene_idx')
    sdf = sdf.withColumnRenamed('row_idx', 'cell_idx')
    return sdf

def read_var_from_h5ads(
    spark:pyspark.sql.session.SparkSession,
    path:Optional[Union[List,str]]=None, 
    df:Optional[pyspark.sql.DataFrame]=None, 
    gene_name_column:Optional[str]=None,
    from_raw:bool=True,
    fallback_default:bool=False,
    force_partitioning: Optional[int]=None,
    ):
    """
    Reads gene metadata from one or more h5ad files into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session to use for DataFrame operations.
    path : str or list of str, optional
        Directory containing h5ad files, a single h5ad file path, or a list of h5ad file paths.
    df : pyspark.sql.DataFrame, optional
        Existing DataFrame containing file paths. If provided, 'path' must be None.
    gene_name_column : str, optional
        Name of the column to use for gene names. If None, only gene IDs are extracted.
    from_raw : bool, optional
        If True, reads gene metadata from the 'raw' group in the h5ad file.
    fallback_default : bool, optional
        If True and 'raw' group is missing, falls back to the default group.
    force_partitioning : int, optional
        If provided, repartitions the DataFrame to the specified number of partitions.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame containing gene metadata with columns: file_path, fp_int, gene_idx, gene_id, gene_name.
    """
    df = construct_h5ad_path_df(path,df,spark)
    
    df = df.select('file_path').distinct()
    df = df.withColumn('fp_int', F.hash('file_path'))

    if force_partitioning:
        df = df.repartition(force_partitioning)

    sdf = df.select('file_path','fp_int')\
        .mapInArrow(
            lambda x: mapinarrow_var_from_h5ad(x, gene_name_column=gene_name_column, from_raw=from_raw, fallback_default=fallback_default), #(gene_name_column),
            schema=genestruct
        )

    sdf = sdf.join(
        df,
        how='left',
        on='fp_int'
    )  
    return sdf

def mapinarrow_obs_from_h5ad(itr: Iterator):
    """
    Extracts cell metadata from h5ad files as Arrow RecordBatches.

    Parameters
    ----------
    itr : Iterator
        Iterator over batches containing file paths and file integer identifiers.
    gene_name_column : str, optional
        Name of the column to use for gene names. If None, only gene IDs are extracted.

    Yields
    ------
    pa.RecordBatch
        Arrow RecordBatch containing fp_int, gene_idx, gene_id, and gene_name for each gene in the file.
    """
    # def mapinarrow_var_from_h5ad(itr: Iterator):
    for batch in itr:
        d = batch.to_pydict()
        for file_path,fp_int in zip(d['file_path'], d['fp_int']):
            file = h5py.File(file_path, 'r')
            
            # parse obs from h5 to dict
            tmp_dict = h5_group_pdf_to_dict(file['obs'], keys = [file['obs'].attrs['_index']]) 
            # convert to df
            tmp_df = pd.DataFrame(tmp_dict)

            tmp_df = tmp_df.rename(columns={file['obs'].attrs['_index']:'cell_barcode'})
            
            logging.info(tmp_df.head())
            
            yield pa.RecordBatch.from_pydict({
                'fp_int': [fp_int]*len(tmp_df),
                'cell_idx' : np.arange(len(tmp_df)),
                'cell_barcode': tmp_df['cell_barcode'],
            }, schema=pa_cell_schema)

def read_obs_from_h5ads(
    spark:pyspark.sql.session.SparkSession,
    path:Optional[Union[List,str]]=None, 
    df:Optional[pyspark.sql.DataFrame]=None, 
    from_raw:bool=True,
    fallback_default:bool=False,
    force_partitioning: Optional[int]=None,
    ):
    """
    Reads gene metadata from one or more h5ad files into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session to use for DataFrame operations.
    path : str or list of str, optional
        Directory containing h5ad files, a single h5ad file path, or a list of h5ad file paths.
    df : pyspark.sql.DataFrame, optional
        Existing DataFrame containing file paths. If provided, 'path' must be None.
    gene_name_column : str, optional
        Name of the column to use for gene names. If None, only gene IDs are extracted.
    from_raw : bool, optional
        If True, reads gene metadata from the 'raw' group in the h5ad file.
    fallback_default : bool, optional
        If True and 'raw' group is missing, falls back to the default group.
    force_partitioning : int, optional
        If provided, repartitions the DataFrame to the specified number of partitions.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame containing gene metadata with columns: file_path, fp_int, gene_idx, gene_id, gene_name.
    """
    df = construct_h5ad_path_df(path,df,spark)
    
    df = df.select('file_path').distinct()
    df = df.withColumn('fp_int', F.hash('file_path'))

    if force_partitioning:
        df = df.repartition(force_partitioning)

    sdf = df.select('file_path','fp_int')\
        .mapInArrow(
            lambda x: mapinarrow_obs_from_h5ad(x),
            schema=cellstruct
        )

    sdf = sdf.join(
        df,
        how='left',
        on='fp_int'
    )  
    return sdf