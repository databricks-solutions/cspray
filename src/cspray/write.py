from delta.tables import DeltaTable

def default_writer(df):
    return df.write\
    .format('delta')\
    .mode("overwrite")\
    .option("overwriteSchema", "True")

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
def default_write_fn(spark, df, table_name, partition=False, cluster=False):
    writer = default_writer(df)
    if partition:
        writer = add_sample_partition(writer)
    # if cluster: # cannot aprtitions and cluster...
    #     writer = add_cluster_by(writer)
    writer.saveAsTable(table_name)
    if cluster:
        add_zorder(spark, table_name)

def path_write_fn(spark, df, table_name, partition=False, cluster=False):
    writer = default_writer(df)
    if partition:
        writer = add_sample_partition(writer)
    writer.save(table_name)
    if cluster:
        add_zorder(spark, table_name, path_based=True)

DEFAULT_WRITERS = {
    'databricks': default_write_fn,
    'delta': path_write_fn
}