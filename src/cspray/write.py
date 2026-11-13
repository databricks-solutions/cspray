from delta.tables import DeltaTable

def fp_int_predicate(fp_ints):
    """SQL predicate covering these ``fp_int`` values.

    Contiguous ids collapse to ``BETWEEN``; leftover (usually changed files)
    are ``IN`` / ``=``. Empty input is an error — a blank replaceWhere is
    not a valid no-op.
    """
    ids = sorted({int(x) for x in fp_ints})
    if not ids:
        raise ValueError('replaceWhere needs at least one fp_int')

    ranges = []
    start = prev = ids[0]
    for n in ids[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    ranges.append((start, prev))

    parts = []
    singles = []
    for lo, hi in ranges:
        if lo == hi:
            singles.append(lo)
        else:
            parts.append(f'fp_int BETWEEN {lo} AND {hi}')
    if len(singles) == 1:
        parts.append(f'fp_int = {singles[0]}')
    elif singles:
        parts.append('fp_int IN ({})'.format(', '.join(str(s) for s in singles)))

    if len(parts) == 1:
        return parts[0]
    return ' OR '.join(f'({p})' for p in parts)

def default_writer(df, replace_where=None):
    writer = df.write.format('delta').mode("overwrite")
    if replace_where:
        # mutually exclusive with overwriteSchema
        return writer.option("replaceWhere", replace_where)
    return writer.option("overwriteSchema", "True")

def add_sample_partition(writer):
    return writer.partitionBy("file_path")

def add_clusterby(writer):
    return writer.clusterBy("cell_idx")

def add_zorder(spark, table_name, path_based=False):
    if path_based:
        delta_table = DeltaTable.forPath(spark, table_name)
    else:
        delta_table = DeltaTable.forName(spark, table_name)
    delta_table.optimize().executeZOrderBy("cell_idx")
    return None    

# might like to have a mode switch in sdata (pyspark, databricks)
def default_write_fn(spark, df, table_name, partition=False, cluster=False, replace_where=None):
    writer = default_writer(df, replace_where=replace_where)
    if partition and not replace_where:
        writer = add_sample_partition(writer)
    # if cluster: # cannot aprtitions and cluster...
    #     writer = add_cluster_by(writer)
    writer.saveAsTable(table_name)
    if cluster and not replace_where:
        add_zorder(spark, table_name)

def path_write_fn(spark, df, table_name, partition=False, cluster=False, replace_where=None):
    writer = default_writer(df, replace_where=replace_where)
    if partition and not replace_where:
        writer = add_sample_partition(writer)
    writer.save(table_name)
    if cluster and not replace_where:
        add_zorder(spark, table_name, path_based=True)

DEFAULT_WRITERS = {
    'databricks': default_write_fn,
    'delta': path_write_fn
}
