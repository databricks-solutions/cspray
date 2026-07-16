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


def _gene_pa_schema(include_meta: bool = False):
    """Arrow schema for the var (gene) mapper, optionally with a JSON metadata field."""
    fields = [
        ('fp_int', pa.int32()),
        ('gene_idx', pa.int64()),
        ('gene_id', pa.string()),
        ('gene_name', pa.string()),
    ]
    if include_meta:
        fields.append(('var_json', pa.string()))
    return pa.schema(fields)


def _gene_spark_schema(include_meta: bool = False):
    """Spark schema for the var (gene) mapper, optionally with a JSON metadata field."""
    fields = [
        T.StructField("fp_int", T.IntegerType(), False),
        T.StructField("gene_idx", T.LongType(), False),
        T.StructField("gene_id", T.StringType(), False),
        T.StructField("gene_name", T.StringType(), False),
    ]
    if include_meta:
        fields.append(T.StructField("var_json", T.StringType(), True))
    return T.StructType(fields)


def _cell_pa_schema(include_meta: bool = False):
    """Arrow schema for the obs (cell) mapper, optionally with a JSON metadata field."""
    fields = [
        ('fp_int', pa.int32()),
        ('cell_idx', pa.int64()),
        ('cell_barcode', pa.string()),
    ]
    if include_meta:
        fields.append(('obs_json', pa.string()))
    return pa.schema(fields)


def _cell_spark_schema(include_meta: bool = False):
    """Spark schema for the obs (cell) mapper, optionally with a JSON metadata field."""
    fields = [
        T.StructField("fp_int", T.IntegerType(), False),
        T.StructField("cell_idx", T.LongType(), False),
        T.StructField("cell_barcode", T.StringType(), False),
    ]
    if include_meta:
        fields.append(T.StructField("obs_json", T.StringType(), True))
    return T.StructType(fields)


def _resolve_wanted_keys(index_key, metadata_columns, always_keep=None):
    """Resolve which h5ad group columns to extract for a given metadata request.

    Parameters
    ----------
    index_key : str
        The group's `_index` column (cell barcode / gene id), always required.
    metadata_columns : None | 'all' | list[str]
        None  -> only the index (+ always_keep): today's behaviour, no metadata.
        'all' -> every column in the group (returns None to signal "all").
        list  -> the index (+ always_keep) plus the requested columns.
    always_keep : list[str], optional
        Extra columns that must be read regardless (e.g. the gene name column).

    Returns
    -------
    list[str] | None
        Ordered, de-duplicated list of keys to read, or None to read all.
    """
    if metadata_columns == 'all':
        return None
    wanted = [index_key]
    if always_keep:
        wanted += [c for c in always_keep if c is not None]
    if metadata_columns is not None:
        wanted += list(metadata_columns)
    # de-duplicate while preserving order
    return list(dict.fromkeys(wanted))


def _rows_to_json(df: pd.DataFrame, double_precision: int = 15):
    """Serialize each row of a DataFrame to a JSON string (one string per row).

    Uses pandas' JSON writer so categoricals, NaN -> null and numpy scalar
    types are handled correctly. `double_precision=15` avoids silently
    truncating float64 values. Columns are expected to be 1-D and aligned to
    the frame length.
    """
    n = len(df)
    if df.shape[1] == 0:
        return ['{}'] * n
    # categoricals -> plain python objects so the values (not integer codes) serialize
    safe = df.copy()
    for c in safe.columns:
        if isinstance(safe[c].dtype, pd.CategoricalDtype):
            safe[c] = safe[c].astype(object)
    json_lines = safe.to_json(orient='records', lines=True, double_precision=double_precision)
    if json_lines == '':
        return ['{}'] * n
    parts = json_lines.split('\n')
    # some pandas versions emit a trailing newline
    if len(parts) == n + 1 and parts[-1] == '':
        parts = parts[:-1]
    return parts



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

    for key in keys:
        if key not in group:
            continue
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


def mapinarrow_var_from_h5ad(itr: Iterator, gene_name_column:Optional[str]=None, from_raw:bool=True, fallback_default:bool=False, metadata_columns:Optional[Union[str,List[str]]]=None):
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
    metadata_columns : 'all' or list[str], optional
        If provided, the remaining var columns are serialized per-gene into a
        JSON string field (`var_json`). 'all' captures every column; a list
        captures the named columns (missing ones are simply skipped per file).
        If None (default), no metadata is captured (original behaviour).

    Yields
    ------
    pa.RecordBatch
        Arrow RecordBatch containing fp_int, gene_idx, gene_id, gene_name, and
        (when metadata_columns is set) a var_json string per gene.
    """
    include_meta = metadata_columns is not None
    for batch in itr:
        d = batch.to_pydict()
        for file_path,fp_int in zip(d['file_path'], d['fp_int']):
            file = h5py.File(file_path, 'r')

            # choose the var group (raw vs default)
            if from_raw:
                if 'raw' in file:
                    group = file['raw']['var']
                elif fallback_default:
                    group = file['var']
                else:
                    raise Exception("No raw data found in h5ad file. Please set fallback_default=True to use the default data instead. Or ensure your h5ad files all have raw if using raw")
            else:
                group = file['var']

            index_key = group.attrs['_index']

            wanted = _resolve_wanted_keys(index_key, metadata_columns, always_keep=[gene_name_column])
            tmp_dict = h5_group_pdf_to_dict(group, keys=wanted)
            tmp_df = pd.DataFrame(tmp_dict)

            n = len(tmp_df)
            ensembl_ids = tmp_df[index_key].values
            if gene_name_column is not None:
                gene_names = tmp_df[gene_name_column].values
            else:
                gene_names = ensembl_ids
            logging.info(tmp_df.head())

            record = {
                'fp_int': [fp_int]*n,
                'gene_idx': np.arange(n),
                'gene_id': ensembl_ids,
                'gene_name': gene_names,
            }
            if include_meta:
                meta_df = tmp_df.drop(columns=[index_key])
                record['var_json'] = _rows_to_json(meta_df)

            yield pa.RecordBatch.from_pydict(record, schema=_gene_pa_schema(include_meta))

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
    metadata_columns:Optional[Union[str,List[str]]]=None,
    metadata_variant:bool=True,
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
    metadata_columns : 'all' or list[str], optional
        If provided, the remaining var columns are captured into a `var_data`
        metadata column (see metadata_variant). None (default) keeps the
        original behaviour of only extracting gene_id / gene_name.
    metadata_variant : bool, optional
        If True (default) the captured metadata is parsed into a VARIANT column
        (`parse_json`, requires Spark 3.5+/DBR 15.3+). If False the metadata is
        left as a JSON string column (portable to OSS Spark). Only relevant when
        metadata_columns is set.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame containing gene metadata with columns: file_path, fp_int,
        gene_idx, gene_id, gene_name, and (when metadata_columns is set) var_data.
    """
    df = construct_h5ad_path_df(path,df,spark)
    
    df = df.select('file_path').distinct()
    df = df.withColumn('fp_int', F.hash('file_path'))

    if force_partitioning:
        df = df.repartition(force_partitioning)

    include_meta = metadata_columns is not None
    sdf = df.select('file_path','fp_int')\
        .mapInArrow(
            lambda x: mapinarrow_var_from_h5ad(x, gene_name_column=gene_name_column, from_raw=from_raw, fallback_default=fallback_default, metadata_columns=metadata_columns),
            schema=_gene_spark_schema(include_meta)
        )

    sdf = sdf.join(
        df,
        how='left',
        on='fp_int'
    )
    if include_meta:
        if metadata_variant:
            sdf = sdf.withColumn('var_data', F.expr('parse_json(var_json)')).drop('var_json')
        else:
            sdf = sdf.withColumnRenamed('var_json', 'var_data')
    return sdf

def mapinarrow_obs_from_h5ad(itr: Iterator, metadata_columns:Optional[Union[str,List[str]]]=None):
    """
    Extracts cell metadata from h5ad files as Arrow RecordBatches.

    Parameters
    ----------
    itr : Iterator
        Iterator over batches containing file paths and file integer identifiers.
    metadata_columns : 'all' or list[str], optional
        If provided, the remaining obs columns are serialized per-cell into a
        JSON string field (`obs_json`). 'all' captures every column; a list
        captures the named columns (missing ones are simply skipped per file).
        If None (default), no metadata is captured (original behaviour).

    Yields
    ------
    pa.RecordBatch
        Arrow RecordBatch containing fp_int, cell_idx, cell_barcode, and
        (when metadata_columns is set) an obs_json string per cell.
    """
    include_meta = metadata_columns is not None
    for batch in itr:
        d = batch.to_pydict()
        for file_path,fp_int in zip(d['file_path'], d['fp_int']):
            file = h5py.File(file_path, 'r')

            index_key = file['obs'].attrs['_index']

            wanted = _resolve_wanted_keys(index_key, metadata_columns)
            tmp_dict = h5_group_pdf_to_dict(file['obs'], keys=wanted)
            tmp_df = pd.DataFrame(tmp_dict)

            n = len(tmp_df)
            logging.info(tmp_df.head())

            record = {
                'fp_int': [fp_int]*n,
                'cell_idx': np.arange(n),
                'cell_barcode': tmp_df[index_key].values,
            }
            if include_meta:
                meta_df = tmp_df.drop(columns=[index_key])
                record['obs_json'] = _rows_to_json(meta_df)

            yield pa.RecordBatch.from_pydict(record, schema=_cell_pa_schema(include_meta))

def read_obs_from_h5ads(
    spark:pyspark.sql.session.SparkSession,
    path:Optional[Union[List,str]]=None, 
    df:Optional[pyspark.sql.DataFrame]=None, 
    from_raw:bool=True,
    fallback_default:bool=False,
    force_partitioning: Optional[int]=None,
    metadata_columns:Optional[Union[str,List[str]]]=None,
    metadata_variant:bool=True,
    ):
    """
    Reads cell metadata from one or more h5ad files into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session to use for DataFrame operations.
    path : str or list of str, optional
        Directory containing h5ad files, a single h5ad file path, or a list of h5ad file paths.
    df : pyspark.sql.DataFrame, optional
        Existing DataFrame containing file paths. If provided, 'path' must be None.
    from_raw : bool, optional
        If True, reads gene metadata from the 'raw' group in the h5ad file.
    fallback_default : bool, optional
        If True and 'raw' group is missing, falls back to the default group.
    force_partitioning : int, optional
        If provided, repartitions the DataFrame to the specified number of partitions.
    metadata_columns : 'all' or list[str], optional
        If provided, the remaining obs columns are captured into an `obs_data`
        metadata column (see metadata_variant). None (default) keeps the
        original behaviour of only extracting the cell barcode.
    metadata_variant : bool, optional
        If True (default) the captured metadata is parsed into a VARIANT column
        (`parse_json`, requires Spark 3.5+/DBR 15.3+). If False the metadata is
        left as a JSON string column (portable to OSS Spark). Only relevant when
        metadata_columns is set.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame containing cell metadata with columns: file_path, fp_int,
        cell_idx, cell_barcode, and (when metadata_columns is set) obs_data.
    """
    df = construct_h5ad_path_df(path,df,spark)
    
    df = df.select('file_path').distinct()
    df = df.withColumn('fp_int', F.hash('file_path'))

    if force_partitioning:
        df = df.repartition(force_partitioning)

    include_meta = metadata_columns is not None
    sdf = df.select('file_path','fp_int')\
        .mapInArrow(
            lambda x: mapinarrow_obs_from_h5ad(x, metadata_columns=metadata_columns),
            schema=_cell_spark_schema(include_meta)
        )

    sdf = sdf.join(
        df,
        how='left',
        on='fp_int'
    )
    if include_meta:
        if metadata_variant:
            sdf = sdf.withColumn('obs_data', F.expr('parse_json(obs_json)')).drop('obs_json')
        else:
            sdf = sdf.withColumnRenamed('obs_json', 'obs_data')
    return sdf