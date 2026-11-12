"""Directory ingest: list, diff against ``sam``, write only the batch.

User entry is ``cspray.ingest.run(spark, path=..., table_base=...)``.
``from_h5ads`` stays the in-memory / overwrite path.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pyspark
from pyspark.sql import functions as F

from .data import SprayData
from .read import DEFAULT_READERS, construct_h5ad_path_df
from .write import fp_int_predicate

logger = logging.getLogger(__name__)

Force = Union[str, int, List[Union[str, int]]]


def _normalize_force(force: Optional[Force]) -> Tuple[List[str], List[int]]:
    """Turn ``force=`` into ``(paths, fp_ints)``. A single value is wrapped.

    ``bool`` is rejected because it is an ``int`` subclass (``True`` would
    become ``fp_int=1``).
    """
    if force is None:
        return [], []
    if isinstance(force, bool):
        raise ValueError('force= must be paths or fp_ints, not bool')
    if isinstance(force, (str, int)):
        items = [force]
    else:
        items = list(force)
    paths, ids = [], []
    for x in items:
        if isinstance(x, bool):
            raise ValueError('force= must be paths or fp_ints, not bool')
        if isinstance(x, str):
            paths.append(x)
        else:
            ids.append(int(x))
    return paths, ids


def diff(
    listing: pyspark.sql.DataFrame,
    sam: Optional[pyspark.sql.DataFrame] = None,
    force: Optional[Force] = None,
) -> Tuple[pyspark.sql.DataFrame, Dict[str, int]]:
    """Split a directory listing against the ``sam`` registry.

    Returns ``(batch, stats)``. ``batch`` is the listing rows to open and
    write (new + changed + forced). Removed files are not in the listing, so
    they are not in the batch — ``if_removed="keep"`` is "do not emit them".
    """
    force_paths, force_ids = _normalize_force(force)
    listing = listing.dropDuplicates(['file_path'])

    if sam is None:
        n = listing.count()
        return listing, {
            'n_new': n, 'n_changed': 0, 'n_skipped': 0, 'n_forced': 0,
        }

    sam_sel = sam.select(
        F.col('file_path'),
        F.col('fp_int').cast('int').alias('_fp_int'),
        (
            F.col('source_length').cast('long')
            if 'source_length' in sam.columns
            else F.lit(None).cast('long')
        ).alias('_sam_length'),
        (
            F.col('source_modified').cast('double')
            if 'source_modified' in sam.columns
            else F.lit(None).cast('double')
        ).alias('_sam_modified'),
    )
    joined = listing.join(F.broadcast(sam_sel), on='file_path', how='left')
    is_new = F.col('_fp_int').isNull()
    if 'source_length' in listing.columns and 'source_modified' in listing.columns:
        is_changed = (~is_new) & ~(
            F.col('source_length').eqNullSafe(F.col('_sam_length'))
            & F.col('source_modified').eqNullSafe(F.col('_sam_modified'))
        )
    else:
        is_changed = F.lit(False)

    is_forced = F.lit(False)
    if force_paths:
        is_forced = is_forced | F.col('file_path').isin(force_paths)
    if force_ids:
        is_forced = is_forced | F.col('_fp_int').isin(force_ids)

    take = is_new | is_changed | is_forced
    counts = joined.select(
        is_new.cast('int').alias('new'),
        ((~is_new) & is_changed).cast('int').alias('changed'),
        ((~is_new) & (~is_changed) & is_forced).cast('int').alias('forced'),
        (~take).cast('int').alias('skipped'),
    ).agg(
        F.coalesce(F.sum('new'), F.lit(0)).alias('n_new'),
        F.coalesce(F.sum('changed'), F.lit(0)).alias('n_changed'),
        F.coalesce(F.sum('forced'), F.lit(0)).alias('n_forced'),
        F.coalesce(F.sum('skipped'), F.lit(0)).alias('n_skipped'),
    ).collect()[0]

    drop = [c for c in ('_fp_int', '_sam_length', '_sam_modified') if c in joined.columns]
    batch = joined.filter(take).drop(*drop)
    stats = {
        'n_new': int(counts.n_new),
        'n_changed': int(counts.n_changed),
        'n_skipped': int(counts.n_skipped),
        'n_forced': int(counts.n_forced),
    }
    return batch, stats


def _load_sam(spark, table_base: str, join_char: str, mode: str):
    read_fn = DEFAULT_READERS[mode]
    name = table_base + join_char + 'sam'
    try:
        df = read_fn(spark, name)
    except Exception:
        return None
    if df.limit(1).count() == 0:
        return None
    return df


def run(
    spark: pyspark.sql.session.SparkSession,
    path: Optional[Union[str, List[str]]] = None,
    df: Optional[pyspark.sql.DataFrame] = None,
    table_base: Optional[str] = 'default',
    join_char: Optional[str] = '.',
    mode: Optional[str] = None,
    force: Optional[Force] = None,
    if_changed: str = 'replace',
    if_removed: str = 'keep',
    **reader_kw: Any,
) -> Dict[str, Any]:
    """Ingest h5ads under ``path`` into ``table_base``.

    Same call every day. Files not yet in ``sam`` are always ingested. Files
    whose size and last-modified time still match ``sam`` are skipped, unless
    you name them in ``force``.

    Day 1 (no ``sam`` table yet): writes all listed files (overwrite).
    Later days: opens only the batch and replaces those files' rows.

    parameters:
    -----------
    spark: Spark session
    path: directory of h5ads, a single h5ad, or a list of h5ad paths (or pass df)
    df: listing frame with at least ``file_path`` (optional if path is set)
    table_base: prefix for the lake tables (catalog.schema, or a directory in delta mode)
    join_char: joiner between table_base and the table name (``.`` or ``/``)
    mode: ``databricks`` (Unity Catalog tables) or ``delta`` (path). Default databricks
    force: which already-ingested files to re-read even though they look unchanged
        (same size and last-modified time as in ``sam``).
        None (default): skip those; only new files and files that look changed.
        Pass a ``file_path``, an ``fp_int``, or a list of those::

            force='/data/a.h5ad'
            force=7
            force=[7, '/data/b.h5ad']

        For a full reload, drop the tables (or ``from_h5ads`` + overwrite) and
        call ``run`` again. Do not pass every file to ``force``.
    if_changed: only ``replace`` is implemented (re-write files whose size or
        last-modified time changed)
    if_removed: only ``keep`` is implemented (files gone from the directory stay in the lake)
    **reader_kw: forwarded to ``from_h5ads`` (e.g. from_raw, chunk_size). ``existing_sam``
        is ignored; this function loads it from the ``sam`` table.
    """
    if mode is None:
        mode = 'databricks'
    if if_changed != 'replace':
        raise ValueError("if_changed other than 'replace' is not implemented")
    if if_removed != 'keep':
        raise ValueError("if_removed other than 'keep' is not implemented")
    reader_kw.pop('existing_sam', None)

    listing = construct_h5ad_path_df(path, df, spark)
    sam = _load_sam(spark, table_base, join_char, mode)
    first_write = sam is None
    batch, stats = diff(listing, sam, force=force)
    stats = dict(stats)
    stats['wrote'] = False

    if batch.limit(1).count() == 0:
        logger.info('ingest: nothing to write %s', stats)
        return stats

    sdata = SprayData.from_h5ads(
        spark,
        df=batch,
        existing_sam=sam,
        mode=mode,
        **reader_kw,
    )
    ids = [r.fp_int for r in sdata.sam.select('fp_int').collect()]
    replace_where = None if first_write else fp_int_predicate(ids)
    sdata.to_tables(
        spark,
        table_base=table_base,
        join_char=join_char,
        replace_where=replace_where,
    )
    stats['wrote'] = True
    stats['fp_ints'] = ids
    logger.info('ingest: wrote %s', stats)
    return stats
