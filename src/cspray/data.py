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
    read_obs_from_h5ads
)
from typing import Optional, List, Dict, Any, Callable, Union
import pyspark
from pyspark.sql import functions as F
import time
import warnings

from .write import DEFAULT_WRITERS
from .read import DEFAULT_READERS
from .utils import get_storage_level

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
        self._file_mapping = None  # Cache for lazy loading
        self.write_order = [
            'X','var','obs','sam','clu','sta'
        ]
        self.set_intermediary_persistance()
        if mode is None:
            mode = 'databricks'
        self.mode=mode

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
            # Try to get from var (smallest table with both columns)
            if self.var is not None:
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
        mode:Optional[str]=None
    ):
        """ Read h5ad files into SprayData object 

        Internally pyspark dataframes are populated with expression, cell, and gene info
        Some assumptions are made about gene/cell columns and because of only loose schema 
        enforcement of h5ad files we only extract the key information for processing
        - additional information from obs can always be processed out and joined in sperately 
        later (if you have some schema in your data) as those files are significantly smaller. 

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
        """


        X = read_expression_from_h5ads(
            spark,
            path=path,
            df=df,
            force_partitioning=force_partitioning,
            chunk_size=chunk_size,
            from_raw=from_raw,
            fallback_default=fallback_default
        )
        var = read_var_from_h5ads(
            spark,
            path=path,
            df=df,
            gene_name_column=gene_name_column,
            from_raw=from_raw,
            fallback_default=fallback_default,
            force_partitioning=force_partitioning,
        )
        
        obs = read_obs_from_h5ads(
            spark,
            path=path,
            df=df,
            force_partitioning=force_partitioning,
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
            If True, partitions tables on file_path. Defaults to True. Only applies to the bigger tables with cell idx: X,obs
        cluster : bool, optional
            If True, clusters tables on cell_idx. Defaults to True.  Only applies to the bigger tables with cell idx: X,obs

        Returns
        -------
        out_subset : list
            List of DataFrame names that were written to tables.
        """
        if write_fn is None:
            write_fn = DEFAULT_WRITERS[self.mode]
        
        if subset is not None:
            subset = sorted(subset, key=lambda x: self.write_order.index(x))
        else:
            subset = self.write_order

        print("subset = ", subset)
        out_subset = []
        for k in subset:
            t0 = time.time()
            if isinstance(self.__dict__[k],pyspark.sql.DataFrame):
                print(table_base+join_char+k)
                if k in ['X','obs']:
                    write_fn(spark, self.__dict__[k], table_base+join_char+k, partition=partition, cluster=cluster)
                else:
                    write_fn(spark, self.__dict__[k], table_base+join_char+k, partition=False, cluster=False)
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
            cluster=cluster
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

    