import pyarrow as pa
import numpy as np
from pyspark.sql import types as T
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats._axis_nan_policy import SmallSampleWarning
import warnings

def get_marker_stats_pa_fn(test='ttest_ind'):
    """
    Returns a function to compute p-values and log2 fold changes for marker genes across clusters.

    Parameters:
        test (str): Statistical test to use ('ttest_ind' or 'mannwhitneyu').

    Returns:
        Callable[[pa.Table], pa.Table]: Function that takes a PyArrow Table and returns a PyArrow Table with p-values, log2 fold changes, and metadata.
    """
    def get_pvalues(table: pa.Table) -> pa.Table:
        df = table.to_pandas()
        cluster_ids = df.cluster_id.drop_duplicates().values
        pval_dicts = []
        with warnings.catch_warnings():
            # some genes may have no cells for testing, handled fine so suppress warning
            
            warnings.simplefilter('ignore', SmallSampleWarning)
            for ci in cluster_ids:
                if test=='mannwhitneyu':
                    result = mannwhitneyu(
                        df[df['cluster_id'] == ci]['log1p_norm_counts'],
                        df[df['cluster_id'] != ci]['log1p_norm_counts'],
                    )
                elif test=="ttest_ind":
                    result = ttest_ind(
                        df[df['cluster_id'] == ci]['log1p_norm_counts'],
                        df[df['cluster_id'] != ci]['log1p_norm_counts'],
                        equal_var=False
                    )
                else:
                    raise ValueError(f"test {test} not supported")
                fc = np.log2(df[df['cluster_id'] == ci]['log1p_norm_counts'].mean()) - np.log2(df[df['cluster_id'] != ci]['log1p_norm_counts'].mean())
                pval_dicts.append({'cluster_id': ci, 'pvalue': result.pvalue, 'fc': fc, 'n1': len(df[df['cluster_id'] == ci])})#, 'n2': len(df[df['cluster_id'] != ci]) })
        out_table = pa.Table.from_pylist(
            pval_dicts,
            schema=pa.schema([
                    pa.field('cluster_id', pa.int32()),
                    pa.field('pvalue', pa.float64()),
                    pa.field('fc', pa.float64()),
                    # pa.field('n1', pa.int32()),
                    # pa.field('n2', pa.int32())
                ])
            )      
        out_table = out_table.append_column("fp_idx", pa.array([df['fp_idx'].iloc[0]]*len(pval_dicts), pa.int32()))
        out_table = out_table.append_column("gene_idx", pa.array([df['gene_idx'].iloc[0]]*len(pval_dicts), pa.int32()))
        return out_table
    return get_pvalues