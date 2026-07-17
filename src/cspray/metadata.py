"""
Metadata profiling and promotion for SprayData obs/var payloads.

When h5ad files are read with ``obs_metadata_columns`` / ``var_metadata_columns``
the flexible, cross-file-heterogeneous obs/var columns are carried through as a
single semi-structured column (``obs_data`` on obs, ``var_data`` on var) — either
a VARIANT (Databricks / Spark 4) or a JSON string (portable / OSS Spark).

This module lets users *understand* that payload — which keys exist, how
consistently they appear across rows/files, their cardinality and value types —
so they can decide what to harmonize and what to ``promote`` into typed
top-level columns for filtering.

Public API (exposed as ``cs.md``):
- ``profile(sdata, which='obs'|'var', per_file=..., ...)`` -> pandas report
- ``profile_obs`` / ``profile_var`` -> thin convenience wrappers
- ``promote(sdata, keys, which='obs'|'var', ...)`` -> materialize keys as columns

The implementation is axis-agnostic (obs and var differ only in the source
DataFrame, the payload column, and the row-key) and works on both VARIANT and
JSON-string columns by normalizing everything to JSON text before profiling.
"""
import json
import re
from typing import List, Optional, Tuple, Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

# which -> (SprayData attribute, default payload column, row-key column)
_AXES = {
    "obs": ("obs", "obs_data", "cell_idx"),
    "var": ("var", "var_data", "gene_idx"),
}

_EXPLODE_SCHEMA = T.StructType(
    [
        T.StructField("g", T.StringType()),
        T.StructField("key", T.StringType()),
        T.StructField("val", T.StringType()),
        T.StructField("vtype", T.StringType()),
    ]
)


def _resolve_axis(sdata, which: str):
    """Return (df, payload_column, row_key) for the requested axis."""
    if which not in _AXES:
        raise ValueError(f"which must be one of {list(_AXES)}, got {which!r}")
    attr, default_col, row_key = _AXES[which]
    df = getattr(sdata, attr)
    if df is None:
        raise ValueError(
            f"sdata.{attr} is None — nothing to profile. "
            f"Read with {which}_metadata_columns=... to capture metadata."
        )
    return df, default_col, row_key


def _payload_as_json_expr(df: DataFrame, col: str):
    """Column expression that yields the payload as a JSON string, regardless of
    whether it is stored as VARIANT, a JSON string, or a struct/map."""
    if col not in df.columns:
        raise ValueError(
            f"column {col!r} not found on the DataFrame (columns: {df.columns}). "
            "Was the metadata captured at read time?"
        )
    dtype = df.schema[col].dataType.simpleString()
    if dtype == "string":
        return F.col(col)
    if dtype == "variant":
        # cast(variant as string) renders JSON text on Databricks / Spark 4
        return F.expr(f"cast(`{col}` as string)")
    # struct / map / other -> serialize to JSON
    return F.to_json(F.col(col))


def _py_type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "other"


def _explode_partition(itr):
    """mapInPandas UDF: parse each row's JSON object and emit one
    (group, key, value_string, value_type) record per top-level key present."""
    for pdf in itr:
        gs = pdf["g"].tolist()
        vs = pdf["v"].tolist()
        out_g: List = []
        out_k: List = []
        out_v: List = []
        out_t: List = []
        for i, s in enumerate(vs):
            if s is None:
                continue
            try:
                obj = json.loads(s)
            except (TypeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            g = gs[i]
            for k, val in obj.items():
                t = _py_type(val)
                if t in ("array", "object"):
                    sval = json.dumps(val, sort_keys=True)
                elif t == "null":
                    sval = None
                else:
                    sval = str(val)
                out_g.append(g)
                out_k.append(k)
                out_v.append(sval)
                out_t.append(t)
        yield pd.DataFrame(
            {"g": out_g, "key": out_k, "val": out_v, "vtype": out_t},
            columns=["g", "key", "val", "vtype"],
        )


def _coverage_tag(pct: float) -> str:
    if pct >= 100.0:
        return "consistent"
    if pct >= 90.0:
        return "mostly"
    if pct >= 50.0:
        return "partial"
    return "sparse"


def _normalize_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", k.lower())


def profile(
    sdata,
    which: str = "obs",
    column: Optional[str] = None,
    per_file: bool = False,
    examples: int = 0,
    sample: Optional[float] = None,
    max_categorical: int = 1000,
    promote_min_coverage: float = 90.0,
    id_like_frac: float = 0.9,
    examples_max_card: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Profile the semi-structured obs/var metadata payload.

    Reports, per top-level key: coverage across rows (present / missing /
    percentage / tag), approximate cardinality, the JSON value types observed,
    whether the types conflict, whether the key looks like an alias of another
    key (name collision after normalization), and a ``suggested_action`` to help
    decide what to promote, keep in the payload, harmonize, or drop.

    Parameters
    ----------
    sdata : SprayData
        Object whose obs/var carries the metadata payload.
    which : str
        'obs' (profiles ``obs_data``) or 'var' (profiles ``var_data``).
    column : str, optional
        Override the payload column name (defaults to obs_data / var_data).
    per_file : bool
        If True, also compute a key x file coverage matrix (harmonization view)
        and return it as a second DataFrame. Requires a ``file_path`` column.
    examples : int
        If > 0, include up to this many most-frequent example values per key
        (only for keys with cardinality <= ``examples_max_card``).
    sample : float, optional
        Fraction (0-1) to sample before profiling for speed on huge tables.
        Coverage/cardinality become approximate.
    max_categorical : int
        Cardinality at or below which a key is considered a usable categorical
        (candidate for promotion). Above it the key is treated as id-like / free
        text and suggested to stay in the payload.
    promote_min_coverage : float
        Minimum coverage percentage for a low-cardinality key to be suggested
        for promotion (below it, and >=50%, it is 'promote_sparse').
    id_like_frac : float
        A key is treated as id-like if its cardinality >= this fraction of the
        rows in which it appears (e.g. per-cell UUIDs, barcodes).
    examples_max_card : int
        Only collect example values for keys with cardinality <= this.
    seed : int
        Seed used when ``sample`` is set.
    verbose : bool
        Pretty-print a human-readable report in addition to returning the frame.

    Returns
    -------
    pandas.DataFrame
        The per-key summary, sorted by coverage descending then key. If
        ``per_file`` is True, returns a tuple ``(summary, per_file_matrix)``.
    """
    df, default_col, _ = _resolve_axis(sdata, which)
    col = column or default_col
    vexpr = _payload_as_json_expr(df, col)

    have_files = "file_path" in df.columns
    if per_file and not have_files:
        raise ValueError("per_file=True requires a 'file_path' column on the axis DataFrame")

    if sample is not None:
        df = df.sample(withReplacement=False, fraction=sample, seed=seed)

    group_col = F.col("file_path") if (per_file and have_files) else F.lit(None).cast("string")
    payload = df.select(group_col.alias("g"), vexpr.alias("v"))

    exploded = payload.mapInPandas(_explode_partition, schema=_EXPLODE_SCHEMA)

    # Persist for reuse across the summary / per-file / examples passes, but only
    # when the SprayData allows it. Serverless compute rejects cache()/persist()
    # ("PERSIST TABLE is not supported"), so honour persist_intermediaries and
    # simply recompute exploded on those passes when persistence is disabled.
    persisted = getattr(sdata, "persist_intermediaries", False)
    if persisted:
        exploded = exploded.persist(sdata.persist_storage_level)

    try:
        total = df.count()

        agg = exploded.groupBy("key").agg(
            F.count(F.lit(1)).alias("present"),
            F.approx_count_distinct("val").alias("n_distinct"),
            F.array_sort(F.collect_set("vtype")).alias("types"),
        )
        pdf = agg.toPandas()

        if pdf.empty:
            summary = pd.DataFrame(
                columns=[
                    "key", "present", "missing", "coverage_pct", "tag",
                    "n_distinct", "value_types", "type_conflict",
                    "possible_alias", "suggested_action",
                ]
            )
        else:
            pdf["missing"] = total - pdf["present"]
            pdf["coverage_pct"] = (pdf["present"] / total * 100).round(1) if total else 0.0
            pdf["tag"] = pdf["coverage_pct"].apply(_coverage_tag)
            pdf["value_types"] = pdf["types"].apply(
                lambda ts: [t for t in ([] if ts is None else ts) if t != "null"]
            )
            pdf["type_conflict"] = pdf["value_types"].apply(lambda ts: len(set(ts)) > 1)

            norm = pdf["key"].map(_normalize_key)
            norm_counts = norm.value_counts()
            pdf["possible_alias"] = norm.map(norm_counts).fillna(1).astype(int) > 1

            pdf["suggested_action"] = pdf.apply(
                lambda r: _suggest_action(
                    r, max_categorical, promote_min_coverage, id_like_frac
                ),
                axis=1,
            )

            summary = pdf[
                [
                    "key", "present", "missing", "coverage_pct", "tag",
                    "n_distinct", "value_types", "type_conflict",
                    "possible_alias", "suggested_action",
                ]
            ].sort_values(["coverage_pct", "key"], ascending=[False, True]).reset_index(drop=True)

            if examples > 0:
                summary = _attach_examples(
                    summary, exploded, examples, examples_max_card, seed
                )

        per_file_matrix = None
        if per_file:
            per_file_matrix = _per_file_matrix(exploded, df)

        if verbose:
            _print_report(summary, total, which, col, per_file_matrix)

        if per_file:
            return summary, per_file_matrix
        return summary
    finally:
        if persisted:
            exploded.unpersist()


def _suggest_action(row, max_categorical, promote_min_coverage, id_like_frac) -> str:
    present = int(row["present"])
    nd = int(row["n_distinct"])
    cov = float(row["coverage_pct"])
    if present == 0:
        return "empty"
    if nd <= 1:
        return "drop_constant" if cov >= 100.0 else "low_signal"
    id_like = nd > max_categorical or (present > 50 and nd >= id_like_frac * present)
    if id_like:
        return "keep_in_variant"
    if cov >= promote_min_coverage:
        return "promote"
    if cov >= 50.0:
        return "promote_sparse"
    return "review_sparse"


def _attach_examples(summary, exploded, examples, examples_max_card, seed):
    low_keys = summary.loc[
        summary["n_distinct"] <= examples_max_card, "key"
    ].tolist()
    if not low_keys:
        summary["examples"] = [[] for _ in range(len(summary))]
        return summary

    counts = (
        exploded.filter(F.col("key").isin(low_keys) & F.col("val").isNotNull())
        .groupBy("key", "val")
        .agg(F.count(F.lit(1)).alias("c"))
    )
    w = Window.partitionBy("key").orderBy(F.desc("c"), F.asc("val"))
    top = (
        counts.withColumn("r", F.row_number().over(w))
        .filter(F.col("r") <= examples)
        .groupBy("key")
        .agg(F.collect_list("val").alias("examples"))
        .toPandas()
    )
    ex_map = dict(zip(top["key"], top["examples"]))
    summary["examples"] = summary["key"].map(lambda k: ex_map.get(k, []))
    return summary


def _per_file_matrix(exploded, df) -> pd.DataFrame:
    file_counts = df.groupBy("file_path").agg(F.count(F.lit(1)).alias("n_rows"))
    pf = (
        exploded.groupBy("g", "key")
        .agg(F.count(F.lit(1)).alias("present"))
        .join(file_counts.withColumnRenamed("file_path", "g"), on="g", how="inner")
        .withColumn("coverage_pct", F.round(F.col("present") / F.col("n_rows") * 100, 1))
        .select(F.col("g").alias("file_path"), "key", "coverage_pct")
        .toPandas()
    )
    if pf.empty:
        return pf
    matrix = (
        pf.pivot(index="key", columns="file_path", values="coverage_pct")
        .fillna(0.0)
    )
    matrix = matrix.reindex(
        matrix.mean(axis=1).sort_values(ascending=False).index
    )
    return matrix


def _print_report(summary, total, which, col, per_file_matrix):
    print(f"\n=== {which} metadata profile: '{col}' over {total:,} rows ===")
    if summary.empty:
        print("  No metadata keys found (payload empty or not captured).")
        return

    consistent = sorted(summary.loc[summary["coverage_pct"] >= 100.0, "key"].tolist())
    sparse = summary.loc[summary["coverage_pct"] < 100.0]

    print(f"\n  Consistent keys — present in ALL {total:,} rows ({len(consistent)}):")
    print(f"    {consistent}")

    print(f"\n  Sparse keys — missing from some rows ({len(sparse)}):")
    for _, r in sparse.sort_values("coverage_pct", ascending=False).iterrows():
        print(
            f"    {r['key']:<40} {r['coverage_pct']:>6.1f}%  "
            f"(missing {int(r['missing']):,} rows)  [{r['tag']}]"
        )

    action_counts = summary["suggested_action"].value_counts().to_dict()
    print(f"\n  Suggested actions: {action_counts}")
    promote = sorted(summary.loc[summary["suggested_action"] == "promote", "key"].tolist())
    if promote:
        print(f"    promote ({len(promote)}): {promote}")
    aliases = sorted(summary.loc[summary["possible_alias"], "key"].tolist())
    if aliases:
        print(f"    possible aliases (name collisions to harmonize): {aliases}")
    conflicts = sorted(summary.loc[summary["type_conflict"], "key"].tolist())
    if conflicts:
        print(f"    type conflicts across rows/files: {conflicts}")

    if per_file_matrix is not None and not per_file_matrix.empty:
        print(f"\n  Per-file coverage matrix ({per_file_matrix.shape[1]} files):")
        print(per_file_matrix.to_string())


def profile_obs(sdata, **kwargs):
    """Convenience wrapper for ``profile(sdata, which='obs', ...)``."""
    return profile(sdata, which="obs", **kwargs)


def profile_var(sdata, **kwargs):
    """Convenience wrapper for ``profile(sdata, which='var', ...)``."""
    return profile(sdata, which="var", **kwargs)


def promote(
    sdata,
    keys: Union[str, List[str]],
    which: str = "obs",
    column: Optional[str] = None,
    prefix: str = "",
    dtypes: Optional[Union[str, dict]] = None,
    inplace: bool = True,
):
    """Materialize selected metadata keys into typed top-level columns.

    Extracts the given keys from the semi-structured payload into their own
    columns on obs/var so they can be used for filtering/joins.

    Parameters
    ----------
    sdata : SprayData
    keys : str or list of str
        Top-level key(s) to promote.
    which : str
        'obs' or 'var'.
    column : str, optional
        Override payload column name (defaults to obs_data / var_data).
    prefix : str
        Prefix for the new column names (default '' -> column named as the key).
    dtypes : str or dict, optional
        Target Spark type for extracted columns. A single string applies to all
        keys; a dict maps key -> type. Defaults to string. For VARIANT payloads
        the type is applied via ``variant_get``; for JSON strings the extracted
        string is cast.
    inplace : bool
        If True (default) update ``sdata.<which>`` in place and return sdata;
        otherwise return the new DataFrame.

    Returns
    -------
    SprayData or pyspark.sql.DataFrame
    """
    if isinstance(keys, str):
        keys = [keys]
    df, default_col, _ = _resolve_axis(sdata, which)
    col = column or default_col
    if col not in df.columns:
        raise ValueError(f"column {col!r} not found on sdata.{which}")
    is_variant = df.schema[col].dataType.simpleString() == "variant"

    def _dtype_for(k):
        if dtypes is None:
            return None
        if isinstance(dtypes, str):
            return dtypes
        return dtypes.get(k)

    exprs = []
    for k in keys:
        target = _dtype_for(k)
        if is_variant:
            tstr = target or "string"
            e = F.expr(f"variant_get(`{col}`, '$.{k}', '{tstr}')")
        else:
            e = F.get_json_object(F.col(col), f"$.{k}")
            if target:
                e = e.cast(target)
        exprs.append(e.alias(f"{prefix}{k}"))

    out = df.select("*", *exprs)
    if inplace:
        setattr(sdata, which, out)
        return sdata
    return out
