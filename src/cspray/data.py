"""
SprayData: A container and utility module for managing single-cell data using Spark DataFrames in Databricks.

This module provides the SprayData class for reading, storing, and writing single-cell expression matrices,
gene/feature metadata, cell/sample/cluster metadata, and additional statistics, all as Spark DataFrames.
It supports reading from h5ad files, integration with Spark SQL tables, and efficient distributed processing
in Spark.

Functions and classes:
- SprayData: Main class for managing single-cell data as Spark DataFrames.
- from_h5ads: Class method to load data from h5ad files.
- to_tables, from_tables: Methods for saving/loading data to/from Spark SQL tables.
- to_tables_and_reset: Write DataFrames to tables and reload them.
- cache: Cache DataFrames in memory for performance.
"""
from .read import (
    read_expression_from_h5ads,
    read_var_from_h5ads,
    read_obs_from_h5ads,
    construct_h5ad_path_df,
    assign_fp_int,
)
from typing import Optional, List, Dict, Any, Callable, Union
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
import time
import warnings

from .write import DEFAULT_WRITERS
from .read import DEFAULT_READERS
from .utils import get_storage_level

# Identity plus ingest signature. Promote / add_sample_metadata cannot overwrite
# these; they are how a later run diffs the directory against the lake.
SAM_IDENTITY_COLUMNS = ('fp_int', 'file_path')
SAM_INGEST_COLUMNS = ('source_length', 'source_modified', 'ingested_at')
SAM_REGISTRY_COLUMNS = SAM_IDENTITY_COLUMNS + SAM_INGEST_COLUMNS


def sam_from_listing(listing: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """One row per file: fp_int, path, listing size/mtime, and ingest time.

    ``source_length`` / ``source_modified`` are copied from the listing when
    present (path-built listings always have them). A caller-supplied ``df``
    that is only ``file_path`` still gets the columns, as nulls, so ``sam``
    schema is stable. ``ingested_at`` is always now.
    """
    listing = assign_fp_int(listing)
    cols = ['fp_int', 'file_path']
    for c in ('source_length', 'source_modified'):
        if c in listing.columns:
            cols.append(c)
    sam = listing.select(*cols).dropDuplicates(['fp_int', 'file_path'])
    if 'source_length' not in sam.columns:
        sam = sam.withColumn('source_length', F.lit(None).cast(T.LongType()))
    if 'source_modified' not in sam.columns:
        sam = sam.withColumn('source_modified', F.lit(None).cast(T.DoubleType()))
    return sam.withColumn('ingested_at', F.current_timestamp())

class SprayData:
    """
    Container for single-cell data stored as Spark DataFrames.

    Parameters
    ----------
    X : pyspark.sql.DataFrame, optional
        Expression matrix.
    var : pyspark.sql.DataFrame, optional
        Gene/feature metadata.
    obs : pyspark.sql.DataFrame, optional
        Cell metadata.
    sam : pyspark.sql.DataFrame, optional
        Sample metadata.
    clu : pyspark.sql.DataFrame, optional
        Cluster metadata.
    sta : pyspark.sql.DataFrame, optional
        Additional statistics or metadata.
    uns : dict, optional
        Unstructured data.
    mode: str, optional
        Mode for handling data. Can be 'databricks' or 'delta'. Defaults to 'databricks'.
        This mostly impacts reading writing from Unity Catalog vs Delta to a disk location. 
        It mat also effect how data is clustered in tables if clustering is applied, though this is 
        more of a radmap difference than a current one.
    """

    def __init__(
        self, 
        X : Optional[pyspark.sql.DataFrame] = None, 
        var : Optional[pyspark.sql.DataFrame] = None, 
        obs : Optional[pyspark.sql.DataFrame] = None, 
        sam : Optional[pyspark.sql.DataFrame] = None, 
        clu : Optional[pyspark.sql.DataFrame] = None,
        sta : Optional[pyspark.sql.DataFrame] = None,
        uns : Optional[Dict[str,Any]] = None,
        mode : Optional[str] = None
    ):
        self.X = X
        self.var = var
        self.obs = obs
        self.sam = sam
        self.clu = clu
        self.sta = sta
        self.uns = uns
        self._file_mapping = None  # Cache for lazy loading
        self.write_order = [
            'X','var','obs','clu','sta','sam'
        ]
        self.set_intermediary_persistance()
        if mode is None:
            mode = 'databricks'
        self.mode=mode

    def _iter_tables(self, subset=None):
        """``(name, df)`` for DataFrames on this object, in ``write_order``.

        ``subset`` restricts names (still ordered by ``write_order``). Names
        not in ``write_order`` raise. Slots that are not DataFrames are skipped.
        """
        if subset is None:
            names = list(self.write_order)
        else:
            extra = [k for k in subset if k not in self.write_order]
            if extra:
                raise ValueError(f'subset not in write_order: {extra}')
            names = sorted(subset, key=self.write_order.index)
        for k in names:
            df = getattr(self, k, None)
            if isinstance(df, pyspark.sql.DataFrame):
                yield k, df

    def set_intermediary_persistance(
        self,
        persist : Optional[bool] = True,
        storage_level : Optional[str] = 'MEMORY_AND_DISK_DESER',
        materialize_off_route : Optional[bool] = True
    ):
        """
        Set persistence options for intermediary DataFrames. By default persistance is on.

        Parameters
        ----------
        persist : bool, optional
            If True, intermediary DataFrames during individual steps will be persisted in memory
            or on disk to improve performance during repeated access or downstream processing.
            Defaults to True. Note, if you rite your own functions for processing you'll use the 
            if sdata.perist_intermediaries: concept to wrap a persist statement.
        storage_level : str, optional
            Specifies the Spark storage level for persisting DataFrames. Common options include
            'MEMORY_AND_DISK', 'MEMORY_ONLY', etc. Defaults to 'MEMORY_AND_DISK_DESER'.
        materialize_off_route : bool, optional
            If True, materializes DataFrames outside the main computation DAG, typically after writing
            the expression matrix (X), to minimize recomputation and optimize resource usage.
            Defaults to True.
        """
        self.persist_intermediaries = persist
        self.persist_storage_level = get_storage_level(storage_level)
        self.materialize_off_route = materialize_off_route

    @property
    def file_mapping(self):
        """
        Lazy-loaded broadcast-friendly mapping of fp_int to file_path.
        Creates and caches the mapping on first access.
        
        Returns
        -------
        pyspark.sql.DataFrame
            Small DataFrame with columns: fp_int, file_path
        """
        if self._file_mapping is None:
            # sam is the sample identity table and preferred source of truth.
            if self.sam is not None:
                self._file_mapping = self.sam.select('fp_int', 'file_path').distinct()
            elif self.var is not None:
                self._file_mapping = self.var.select('fp_int', 'file_path').distinct()
            elif self.obs is not None:
                self._file_mapping = self.obs.select('fp_int', 'file_path').distinct()
            elif self.X is not None:
                self._file_mapping = self.X.select('fp_int', 'file_path').distinct()
            else:
                raise ValueError("No dataframes available to create file_mapping")
        return self._file_mapping
    
    def invalidate_file_mapping(self):
        """
        Invalidate the cached file mapping. 
        Call this if you modify file_path or fp_int columns.
        """
        self._file_mapping = None

    @classmethod
    def from_h5ads(
        cls, 
        spark:pyspark.sql.session.SparkSession,
        path: Optional[str | List[str]] = None,
        df:Optional[pyspark.sql.DataFrame]=None, 
        force_partitioning: Optional[int]=None,
        chunk_size: Optional[int]=30_000_000,
        broadcast_genes: Optional[bool] = False,
        gene_name_column: Optional[str] = None,
        ensembl_reference_df: Optional[pyspark.sql.DataFrame]= None,
        from_raw:Optional[bool]=True,
        fallback_default:Optional[bool]=False,
        mode:Optional[str]=None,
        obs_metadata_columns: Optional[Union[str, List[str]]] = None,
        var_metadata_columns: Optional[Union[str, List[str]]] = None,
        metadata_variant: Optional[bool] = True,
        metadata_chunk_size: Optional[int] = 50_000,
        existing_sam: Optional[pyspark.sql.DataFrame] = None,
    ):
        """ Read h5ad files into SprayData object 

        Internally pyspark dataframes are populated with expression, cell, and gene info
        Some assumptions are made about gene/cell columns and because of only loose schema 
        enforcement of h5ad files we only extract the key information for processing
        - additional information from obs can always be processed out and joined in sperately 
        later (if you have some schema in your data) as those files are significantly smaller. 

        The flexible, non-key obs/var columns can optionally be carried through as a single
        semi-structured column (`obs_data` on obs, `var_data` on var) via the
        obs_metadata_columns / var_metadata_columns arguments. This sidesteps the
        cross-file schema heterogeneity of h5ad (different files can have different
        obs/var columns) by storing each row's extra columns as a VARIANT (or JSON
        string). The processing pipeline continues to operate on the skinny key
        columns; the metadata column simply rides along on obs/var.

        `sam` is seeded here from the file listing (fp_int, file_path, size,
        mtime) plus ingested_at, not from obs, so writing sam does not
        recompute the cell table from HDF5. Those columns are the ingest
        registry; promote/add_sample_metadata cannot overwrite them.

        parameters:
        -----------
        path: path to dir of h5ad files, or single h5ad file, or list of h5ad file paths (optional - provide path or df)
        df: spark dataframe with `file_path` column (optional, can pass path if no df yet made)
        spark: park session to use for reading files
        force_partioning: optional - number of partitions to force (over chunks) not recommended to use except small scale testing
        chunk_size: number of rows of sparse expression to read at a time
        broadcast_genes: broadcast gene table to workers
        gene_name_column: if provided will use this column to get provided gene names (not ensembl)
        ensembl_reference_df: an optional reference df of ensembl gene ids to gene names, if provided will use instead of h5ad provided gene names
        from_raw: if True will read from raw group in h5ad
        fallback_default: if True will fallback to default read if no raw group (only used if from_raw=True)
        obs_metadata_columns: 'all' or list of obs column names to capture into an `obs_data` column. None (default) captures none.
        var_metadata_columns: 'all' or list of var column names to capture into a `var_data` column. None (default) captures none.
        metadata_variant: if True (default) metadata columns are stored as VARIANT (needs Spark 3.5+/DBR 15.3+); if False they are kept as JSON strings.
        metadata_chunk_size: number of obs/var rows (cells/genes) read per chunk. Bounds per-worker memory independent of file size; files smaller than this read in a single chunk. None reads each file whole. Defaults to 50,000.
        existing_sam: optional registry (at least file_path, fp_int). Hits keep
            their id; new paths get max+1. None (default) numbers the listing
            0, 1, … in path order.
        """


        listing = assign_fp_int(
            construct_h5ad_path_df(path, df, spark),
            existing_sam=existing_sam,
        )
        sam = sam_from_listing(listing)

        X = read_expression_from_h5ads(
            spark,
            df=listing,
            force_partitioning=force_partitioning,
            chunk_size=chunk_size,
            from_raw=from_raw,
            fallback_default=fallback_default
        )
        var = read_var_from_h5ads(
            spark,
            df=listing,
            gene_name_column=gene_name_column,
            from_raw=from_raw,
            fallback_default=fallback_default,
            force_partitioning=force_partitioning,
            metadata_columns=var_metadata_columns,
            metadata_variant=metadata_variant,
            metadata_chunk_size=metadata_chunk_size,
        )
        
        obs = read_obs_from_h5ads(
            spark,
            df=listing,
            force_partitioning=force_partitioning,
            metadata_columns=obs_metadata_columns,
            metadata_variant=metadata_variant,
            metadata_chunk_size=metadata_chunk_size,
        ) # we assume obs always in regular and not raw
        
        if ensembl_reference_df is not None:
            var = var.withColumnRenamed('gene_name','original_provided_gene_name')
            var = var.join(
                F.broadcast(ensembl_reference_df),
                on='gene_id',
                how='left'
            )
            
        # pull gene names into the expression - very useful for downstream analyis
        # optionally broadcast gene table if expected to fit on workers
        if broadcast_genes:
            X = X.join(
                F.broadcast(var.select('file_path','gene_idx','gene_name')),
                on=['file_path','gene_idx'],
                how='left'
            )
        else:
            X = X.join(
                var.select('file_path','gene_idx','gene_name'),
                on=['file_path','gene_idx'],
                how='left'
            )
        return cls(
            X=X,
            var=var,
            obs=obs,
            sam=sam,
            mode=mode
        )

    def to_tables(
        self,
        spark:pyspark.sql.session.SparkSession, 
        write_fn:Optional[Callable]=None, 
        table_base:Optional[str]='default', 
        join_char:Optional[str]='.', 
        subset:Optional[List]=None,
        partition: Optional[bool]=False,
        cluster: Optional[bool]=False,
        replace_where: Optional[str]=None,
        ):
        """
        Write DataFrames in SprayData to tables in a Spark session.

        Parameters
        ----------
        write_fn : Callable, optional
            Function to write DataFrames to tables. Defaults to default_write_fn.
        table_base : str, optional
            Base name for tables. Defaults to 'default'.
        join_char : str, optional
            Character to join base name and table name. Defaults to '.'.
        subset : list, optional
            List of DataFrame names to write. Defaults to all in write_order.
        partition : bool, optional
            If True, partitionBy file_path on tables that have that column.
        cluster : bool, optional
            If True, Z-order by cell_idx on tables that have that column.
        replace_where : str, optional
            Delta predicate (usually from ``fp_int_predicate``). None (default)
            overwrites each table, including schema. Set, it replaces only
            matching rows and cannot change schema.

        Returns
        -------
        out_subset : list
            List of DataFrame names that were written to tables.
        """
        if write_fn is None:
            write_fn = DEFAULT_WRITERS[self.mode]

        print("subset = ", subset if subset is not None else self.write_order)
        out_subset = []
        for k, df in self._iter_tables(subset):
            t0 = time.time()
            print(table_base+join_char+k)
            write_fn(
                spark, df, table_base+join_char+k,
                partition=bool(partition) and 'file_path' in df.columns,
                cluster=bool(cluster) and 'cell_idx' in df.columns,
                replace_where=replace_where,
            )
            out_subset.append(k)
            print("time elapsed = ", time.time()-t0)

        if self.persist_intermediaries:
            spark.catalog.clearCache()
        return out_subset
        
    
    @classmethod
    def from_tables(
        cls,
        spark:pyspark.sql.session.SparkSession,
        table_base:str,
        join_char:str='.',
        read_fn:Optional[Callable]=None,
        mode:Optional[str]=None
    ):
        """
        Load SprayData object from Spark tables.

        Parameters
        ----------
        spark : pyspark.sql.session.SparkSession
            Spark session to use for reading tables.
        table_base : str
            Base name for tables.
        join_char : str, optional
            Character to join base name and table name. Defaults to '.'.

        Returns
        -------
        SprayData
            SprayData object with DataFrames loaded from tables.
        """
        cls_ = cls(mode=mode)

        if read_fn is None:
            read_fn = DEFAULT_READERS[cls_.mode]

        for k in cls_.write_order:
            try:
                cls_.__dict__[k] = read_fn(spark, table_base+join_char+k) #spark.table(table_base+join_char+k)
            except Exception as e:
                warnings.warn(f"table {table_base+join_char+k} not found;\n error was : {e}")
                cls_.__dict__[k] = None
        return cls_
    
    def to_tables_and_reset(
        self,
        spark:pyspark.sql.session.SparkSession,
        write_fn:Optional[Callable]=None, 
        read_fn:Optional[Callable]=None,
        table_base:Optional[str]='default', 
        join_char:Optional[str]='.',
        subset:Optional[List]=None,
        partition: Optional[bool]=False,
        cluster: Optional[bool]=False,
        replace_where: Optional[str]=None,
    ):
        """
        Write DataFrames in SprayData to tables and reset them by reloading from tables.

        Parameters
        ----------
        spark : pyspark.sql.session.SparkSession
            Spark session to use for writing and reading tables.
        write_fn : Callable, optional
            Function to write DataFrames to tables. Defaults to default_write_fn.
        read_fn : Callable, optional
            Function to read DataFrames from tables. Defaults to default_read_fn.
        table_base : str, optional
            Base name for tables. Defaults to 'default'.
        join_char : str, optional
            Character to join base name and table name. Defaults to '.'.
        subset : list, optional
            List of DataFrame names to write and reset. Defaults to all in write_order.
        replace_where : str, optional
            Forwarded to ``to_tables``. None (default) is a full overwrite.
        
        Returns
        -------
        None
        """
        if read_fn is None:
            read_fn = DEFAULT_READERS[self.mode]
        
        subset = self.to_tables(
            spark,
            write_fn=write_fn, 
            table_base=table_base, 
            join_char=join_char, 
            subset=subset, 
            partition=partition,
            cluster=cluster,
            replace_where=replace_where,
        )

        # reset the ones that were saved
        for k in subset:
            try:
                self.__dict__[k] = read_fn(spark, table_base+join_char+k)
            except Exception as e:
                warnings.warn(f"table {table_base+join_char+k} not found;\n error was : {e}")
                self.__dict__[k] = None
        return None

    def to_anndata(self,file_paths:List[str]):
        pass

    