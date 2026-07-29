"""
Metadata profiling and promotion for SprayData obs/var payloads, plus the
sample table (``sam``) that sample-grain metadata is collected onto.

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
- ``promote(sdata, keys, which='obs'|'var', ...)`` -> materialize given keys as columns
- ``promote_suggested(sdata, which='obs'|'var'|'both', promote_sam=...)`` ->
  profile + promote the recommended keys (typed) in one call, optionally
  routing sample-constant keys to ``sam``
- ``add_sample_metadata(sdata, df, on=...)`` -> attach a one-row-per-sample
  table (study design, donor tables) to ``sam``
- ``profile_sample(sdata)`` -> pandas report of which obs keys are constant
  within each input file (the sample grain)
- ``promote_sample(sdata, keys, ...)`` -> attach one typed value per file to
  ``sam`` instead of repeating it on every cell
- ``promote_sample_suggested(sdata, ...)`` -> profile_sample + promote_sample

Promotion offers three paths depending on how much control you want:

1. Automatic  -> ``promote_suggested(sdata, which='both')`` profiles and
   promotes the recommended keys with inferred types in one call.
2. Review-then-apply -> ``promote_suggested(..., dry_run=True)`` returns the
   plan (key -> dtype -> action) to edit, then you call ``promote`` with it.
3. Full manual -> use ``profile`` + ``promote`` directly with your own key list
   and ``dtypes``.

The sample functions mirror those paths (``promote_sample_suggested`` supports
``dry_run``), or ride along with the obs/var pass via
``promote_suggested(..., promote_sam=True)``. All promotion is a post-read step:
read captures the payload, then you decide what to materialize and where it
belongs.

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

# Maps the JSON value types reported by profile() (in the `value_types` column)
# to the Spark type promote() should cast to. Nested / mixed types fall back to
# string. Editable by callers that want different defaults.
DEFAULT_VARIANT_TYPE_MAP = {
    "int": "bigint",
    "float": "double",
    "bool": "boolean",
    "str": "string",
    "object": "string",
    "array": "string",
}


def _spark_type_from_value_types(value_types, type_map=None) -> str:
    """Pick a Spark type for a key from the JSON value types profile() observed.

    Returns the mapped type only when a single (non-null) JSON type was seen;
    mixed types (a type conflict across rows/files) fall back to 'string'.
    """
    tm = type_map or DEFAULT_VARIANT_TYPE_MAP
    non_null = [t for t in (value_types if value_types is not None else []) if t != "null"]
    if len(set(non_null)) == 1:
        return tm.get(non_null[0], "string")
    return "string"


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


def _metadata_value_expr(df: DataFrame, col: str, key: str, dtype: Optional[str] = None):
    """Extract one top-level metadata value from VARIANT or JSON."""
    if df.schema[col].dataType.simpleString() == "variant":
        return F.expr(f"variant_get(`{col}`, '$.{key}', '{dtype or 'string'}')")
    value = F.get_json_object(F.col(col), f"$.{key}")
    return value.cast(dtype) if dtype else value


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


def _merge_sample_metadata(sdata, metadata: DataFrame, on, if_exists: str):
    """Left-join sample-grain columns onto ``sam``, enforcing its invariants.

    Skips the one-row-per-key check, so internal callers passing an
    already-grouped DataFrame avoid an extra Spark action.
    """
    if sdata.sam is None:
        raise ValueError("sdata.sam is None; sample identities must exist before adding metadata")

    keys = [on] if isinstance(on, str) else list(on)
    new_columns = [c for c in metadata.columns if c not in keys]

    # Replacing these would rewrite sample identity from an arbitrary table.
    identity = [c for c in new_columns if c in ("fp_int", "file_path")]
    if identity:
        raise ValueError(f"sample identity columns cannot be joined as metadata: {identity}")

    base = sdata.sam
    conflicts = [c for c in new_columns if c in base.columns]
    if conflicts:
        if if_exists != "replace":
            raise ValueError(
                f"sample metadata columns already exist on sam: {conflicts}; "
                "pass if_exists='replace' to update them"
            )
        base = base.drop(*conflicts)

    sdata.sam = base.join(metadata.select(*keys, *new_columns), on=keys, how="left")
    return sdata


def add_sample_metadata(
    sdata,
    metadata: DataFrame,
    on: Union[str, List[str]] = "file_path",
    if_exists: str = "error",
):
    """Attach a one-row-per-sample DataFrame to ``sam``.

    The entry point for sample annotations that do not come from the h5ad files
    (study design, donor tables, external QC). ``sam`` is seeded at read time
    with one row per input file, so this can be called at any point in a
    workflow, before or after QC and obs metadata promotion.

    Parameters
    ----------
    metadata : pyspark.sql.DataFrame
        Sample-grain DataFrame containing the join key(s) and new columns.
    on : str or list of str
        Column(s) identifying a sample. Defaults to ``file_path``.
    if_exists : str
        ``"error"`` (default) or ``"replace"`` for columns already on ``sam``.

    Returns
    -------
    SprayData
    """
    keys = [on] if isinstance(on, str) else list(on)
    duplicate = (
        metadata.groupBy(*keys).count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )
    if duplicate:
        raise ValueError(f"sample metadata must have at most one row per {keys}")
    return _merge_sample_metadata(sdata, metadata, on=on, if_exists=if_exists)


def profile_sample(
    sdata,
    column: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Classify obs metadata keys by whether they are sample-constant.

    An input file is the current sample grain. A key is suggested for sample
    promotion when it has at most one distinct non-null value within every
    ``file_path`` where it is present. Values may differ across files.

    Parameters
    ----------
    column : str, optional
        Override the payload column name (defaults to obs_data).
    verbose : bool
        Print a short summary in addition to returning the frame.

    Returns
    -------
    pandas.DataFrame
        One row per key with ``n_samples_present``, ``n_samples_total``,
        ``max_values_per_sample``, ``value_types``, ``type_conflict`` and a
        ``suggested_action`` of ``promote_sample`` or ``keep_on_obs``.
    """
    df, default_col, _ = _resolve_axis(sdata, "obs")
    col = column or default_col
    if "file_path" not in df.columns:
        raise ValueError("sample metadata profiling requires file_path on sdata.obs")

    payload = df.select(
        F.col("file_path").alias("g"),
        _payload_as_json_expr(df, col).alias("v"),
    )
    exploded = payload.mapInPandas(_explode_partition, schema=_EXPLODE_SCHEMA)
    persisted = getattr(sdata, "persist_intermediaries", False)
    if persisted:
        exploded = exploded.persist(sdata.persist_storage_level)

    columns = [
        "key", "n_samples_present", "n_samples_total",
        "max_values_per_sample", "value_types", "type_conflict",
        "suggested_action",
    ]
    try:
        total_samples = (
            sdata.sam.select("file_path").distinct().count()
            if sdata.sam is not None
            else df.select("file_path").distinct().count()
        )
        non_null = exploded.filter(F.col("val").isNotNull())
        per_sample = non_null.groupBy("g", "key").agg(
            F.countDistinct("val").alias("n_values")
        )
        summary = per_sample.groupBy("key").agg(
            F.countDistinct("g").alias("n_samples_present"),
            F.max("n_values").alias("max_values_per_sample"),
        ).join(
            non_null.groupBy("key").agg(
                F.array_sort(F.collect_set("vtype")).alias("types")
            ),
            on="key",
            how="left",
        ).toPandas()

        if summary.empty:
            return pd.DataFrame(columns=columns)

        summary["n_samples_total"] = total_samples
        summary["value_types"] = summary["types"].apply(
            lambda ts: [t for t in ([] if ts is None else ts) if t != "null"]
        )
        summary["type_conflict"] = summary["value_types"].apply(
            lambda ts: len(set(ts)) > 1
        )
        summary["suggested_action"] = summary["max_values_per_sample"].apply(
            lambda n: "promote_sample" if int(n) <= 1 else "keep_on_obs"
        )
        result = summary[columns].sort_values(
            ["suggested_action", "key"], ascending=[False, True]
        ).reset_index(drop=True)

        if verbose:
            counts = result["suggested_action"].value_counts().to_dict()
            print(f"sample metadata profile ({total_samples} files): {counts}")
            promote_keys = result.loc[
                result["suggested_action"] == "promote_sample", "key"
            ].tolist()
            if promote_keys:
                print(f"  promote_sample: {promote_keys}")
        return result
    finally:
        if persisted:
            exploded.unpersist()


def promote_sample(
    sdata,
    keys: Union[str, List[str]],
    column: Optional[str] = None,
    prefix: str = "",
    dtypes: Optional[Union[str, dict]] = None,
    strict: bool = True,
    if_exists: str = "replace",
    drop_from_obs: bool = True,
):
    """Attach selected sample-constant obs metadata keys to ``sam``.

    The original semi-structured ``obs_data`` payload is retained. By default,
    an already-promoted top-level obs column with the same name is removed so a
    sample-level value is not duplicated on every cell.

    Parameters
    ----------
    keys : str or list of str
        Top-level obs payload keys to move onto ``sam``.
    column : str, optional
        Override the payload column name (defaults to obs_data).
    prefix : str
        Optional prefix for the created ``sam`` column names.
    dtypes : str or dict, optional
        Single Spark type for all keys, or per-key mapping. None keeps strings.
    strict : bool
        If True (default) raise when a key has more than one distinct value
        within a file rather than silently taking the first.
    if_exists : str
        ``"replace"`` (default) or ``"error"`` for columns already on ``sam``.
    drop_from_obs : bool
        If True (default) drop a same-named top-level obs column so the value
        is not duplicated per cell.

    Returns
    -------
    SprayData
    """
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    if not keys:
        return sdata

    obs, default_col, _ = _resolve_axis(sdata, "obs")
    col = column or default_col
    if col not in obs.columns:
        raise ValueError(f"column {col!r} not found on sdata.obs")

    def _dtype_for(key):
        if isinstance(dtypes, str):
            return dtypes
        if isinstance(dtypes, dict):
            return dtypes.get(key)
        return None

    aliases = [f"__sample_value_{i}" for i in range(len(keys))]
    values = obs.select(
        "file_path",
        *[
            _metadata_value_expr(obs, col, key, _dtype_for(key)).alias(alias)
            for key, alias in zip(keys, aliases)
        ],
    )

    aggregations = []
    for alias in aliases:
        aggregations.extend([
            F.first(F.col(alias), ignorenulls=True).alias(alias),
            F.countDistinct(F.col(alias)).alias(f"{alias}__n_distinct"),
        ])
    by_sample = values.groupBy("file_path").agg(*aggregations)

    if strict:
        conflict_expr = None
        for alias in aliases:
            current = F.col(f"{alias}__n_distinct") > 1
            conflict_expr = current if conflict_expr is None else (conflict_expr | current)
        conflicts = by_sample.filter(conflict_expr).select("file_path").limit(10).collect()
        if conflicts:
            paths = [row["file_path"] for row in conflicts]
            raise ValueError(
                "metadata is not constant within every sample; "
                f"conflicting file paths include {paths}"
            )

    output_names = [f"{prefix}{key}" for key in keys]
    sample_metadata = by_sample.select(
        "file_path",
        *[
            F.col(alias).alias(output_name)
            for alias, output_name in zip(aliases, output_names)
        ],
    )
    _merge_sample_metadata(sdata, sample_metadata, on="file_path", if_exists=if_exists)

    if drop_from_obs:
        existing = [name for name in output_names if name in sdata.obs.columns]
        if existing:
            sdata.obs = sdata.obs.drop(*existing)
    return sdata


def promote_sample_suggested(
    sdata,
    keys: Optional[List[str]] = None,
    infer_types: bool = True,
    dtypes: Optional[Union[str, dict]] = None,
    type_map: Optional[dict] = None,
    prefix: str = "",
    dry_run: bool = False,
    verbose: bool = True,
    drop_from_obs: bool = True,
):
    """Profile obs metadata and promote sample-constant keys onto ``sam``.

    The sample-level counterpart of :func:`promote_suggested`: runs
    :func:`profile_sample`, keeps the keys whose ``suggested_action`` is
    ``promote_sample`` (or the explicit ``keys`` given), infers a Spark type for
    each, and calls :func:`promote_sample`.

    Parameters
    ----------
    keys : list of str, optional
        Explicit keys to promote, bypassing the suggestion filter.
    infer_types : bool
        Infer a Spark type per key from the profiled value types.
    dtypes : str or dict, optional
        Override the inferred types for all / selected keys.
    type_map : dict, optional
        Custom JSON-type to Spark-type mapping (see DEFAULT_VARIANT_TYPE_MAP).
    prefix : str
        Optional prefix for the created ``sam`` column names.
    dry_run : bool
        If True, return the plan (``which, key, dtype, suggested_action``) and
        do NOT modify ``sdata``.
    verbose : bool
        If True (default) print the plan before applying.
    drop_from_obs : bool
        Passed through to :func:`promote_sample`.

    Returns
    -------
    SprayData or pandas.DataFrame
        ``sdata``, or the plan when ``dry_run=True``.
    """
    report = profile_sample(sdata, verbose=False)
    if keys is None:
        selected = report[report["suggested_action"] == "promote_sample"].copy()
    else:
        selected = report[report["key"].isin(keys)].copy()

    def _resolve(row):
        key = row["key"]
        if isinstance(dtypes, str):
            return dtypes
        if isinstance(dtypes, dict) and key in dtypes:
            return dtypes[key]
        if infer_types:
            return _spark_type_from_value_types(row["value_types"], type_map)
        return "string"

    plan = selected[["key", "suggested_action", "value_types"]].copy()
    plan["dtype"] = plan.apply(_resolve, axis=1) if not plan.empty else []
    plan.insert(0, "which", "sample")
    plan = plan[["which", "key", "dtype", "suggested_action"]]

    if verbose or dry_run:
        if plan.empty:
            print("promote_sample_suggested: no keys matched — nothing to promote.")
        else:
            print(f"promote_sample_suggested plan ({len(plan)} columns):")
            print(plan.to_string(index=False))
    if dry_run:
        return plan

    if not plan.empty:
        promote_sample(
            sdata,
            plan["key"].tolist(),
            prefix=prefix,
            dtypes=dict(zip(plan["key"], plan["dtype"])),
            drop_from_obs=drop_from_obs,
        )
    return sdata


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
    columns on obs/var so they can be used for filtering/joins. This is the
    low-level, full-control primitive: you supply the exact keys and (optionally)
    their types. For the common "promote everything worth filtering on, typed"
    workflow, see :func:`promote_suggested`, which profiles and calls this for
    you (and supports a ``dry_run`` review step).

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

    def _dtype_for(k):
        if dtypes is None:
            return None
        if isinstance(dtypes, str):
            return dtypes
        return dtypes.get(k)

    exprs = []
    for k in keys:
        exprs.append(
            _metadata_value_expr(df, col, k, _dtype_for(k)).alias(f"{prefix}{k}")
        )

    out = df.select("*", *exprs)
    if inplace:
        setattr(sdata, which, out)
        return sdata
    return out


def _plan_for_axis(sdata, which, actions, keys, infer_types, dtypes, type_map, profile_kwargs):
    """Build the promotion plan (key -> dtype) for one axis from a profile run."""
    pk = dict(profile_kwargs or {})
    pk["verbose"] = False
    pk.pop("per_file", None)  # a plan needs the flat summary, not the (summary, matrix) tuple
    report = profile(sdata, which=which, **pk)
    if isinstance(report, tuple):  # defensive: per_file slipped through
        report = report[0]

    if keys is not None:
        sel = report[report["key"].isin(list(keys))]
    else:
        sel = report[report["suggested_action"].isin(list(actions))]

    plan = sel[["key", "suggested_action", "value_types"]].copy()

    def _resolve(row):
        if isinstance(dtypes, str):
            return dtypes
        if isinstance(dtypes, dict) and row["key"] in dtypes:
            return dtypes[row["key"]]
        if infer_types:
            return _spark_type_from_value_types(row["value_types"], type_map)
        return "string"

    plan["dtype"] = plan.apply(_resolve, axis=1) if not plan.empty else []
    plan.insert(0, "which", which)
    return plan[["which", "key", "dtype", "suggested_action"]]


def promote_suggested(
    sdata,
    which: str = "obs",
    actions: Tuple[str, ...] = ("promote",),
    keys: Optional[List[str]] = None,
    infer_types: bool = True,
    dtypes: Optional[Union[str, dict]] = None,
    type_map: Optional[dict] = None,
    prefix: str = "",
    profile_kwargs: Optional[dict] = None,
    promote_sam: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Profile, then promote the recommended keys as typed columns — in one call.

    This is the high-level convenience path for the common workflow of
    "materialize the metadata keys worth filtering on, with sensible types".
    It runs :func:`profile`, selects the keys whose ``suggested_action`` is in
    ``actions``, infers a Spark type for each from the profiled ``value_types``
    (via :data:`DEFAULT_VARIANT_TYPE_MAP`), and calls :func:`promote`.

    There are three paths, in increasing order of control — pick the one that
    matches how much you want to intervene:

    1. One call, fully automatic (types inferred, both axes if desired, and
       ``promote_sam=True`` to also collect sample-constant keys onto ``sam``)::

        cs.md.promote_suggested(sdata, which='both', promote_sam=True)

    2. Review-then-apply: get the plan, edit it, apply it yourself. Use
       ``dry_run=True`` to return the plan (key -> dtype -> action) WITHOUT
       mutating ``sdata``, then hand your edited selection to :func:`promote`::

        plan = cs.md.promote_suggested(sdata, which='obs', dry_run=True)
        plan = plan[plan.key != 'unwanted_key']
        cs.md.promote(sdata, plan.key.tolist(),
                      dtypes=dict(zip(plan.key, plan.dtype)), which='obs')

    3. Full manual control: skip this function and use :func:`profile` +
       :func:`promote` directly, supplying your own key list and ``dtypes``.

    Parameters
    ----------
    sdata : SprayData
    which : str
        'obs', 'var', or 'both' (loops obs then var). ``keys`` is not allowed
        with 'both'.
    actions : tuple of str
        Which ``suggested_action`` values to promote. Defaults to
        ``('promote',)``; pass ``('promote', 'promote_sparse')`` to also include
        partial-coverage categoricals. Applies to the obs/var report only - the
        sample side has its own vocabulary and is always taken as suggested.
    keys : list of str, optional
        Explicit keys to promote, bypassing the ``actions`` filter. Only valid
        for a single axis (not 'both').
    infer_types : bool
        If True (default) infer each column's Spark type from the profiled
        value types; if False everything is promoted as string (unless
        overridden by ``dtypes``).
    dtypes : str or dict, optional
        Explicit type override(s): a single string for all keys, or a
        ``key -> type`` dict (takes precedence over inference for those keys).
    type_map : dict, optional
        Override the JSON-type -> Spark-type mapping (defaults to
        :data:`DEFAULT_VARIANT_TYPE_MAP`).
    prefix : str
        Prefix for the promoted column names (guards against collisions with
        existing columns).
    profile_kwargs : dict, optional
        Extra keyword args forwarded to :func:`profile` (e.g. ``max_categorical``,
        ``sample``). ``verbose`` and ``per_file`` are managed internally.
    promote_sam : bool
        If True, also run :func:`promote_sample_suggested`: keys that are
        constant within each input file go to ``sam`` instead of being repeated
        as an obs column. Those keys are dropped from the obs plan, so a key
        only ever lands in one place. Off by default; adds a second profiling
        pass over the obs payload.
    dry_run : bool
        If True, return the promotion plan (a pandas DataFrame with columns
        ``which, key, dtype, suggested_action``) and do NOT modify ``sdata``.
    verbose : bool
        If True (default) print the plan before applying.

    Returns
    -------
    SprayData or pandas.DataFrame
        The updated ``sdata`` (when applied), or the plan DataFrame (when
        ``dry_run=True``).

    See Also
    --------
    profile : the read-only report this builds on.
    promote : the low-level, full-control materialization primitive.
    """
    if which not in ("obs", "var", "both"):
        raise ValueError(f"which must be 'obs', 'var', or 'both', got {which!r}")
    if keys is not None and which == "both":
        raise ValueError("keys= is only supported for a single axis, not which='both'")
    if keys is not None and promote_sam:
        raise ValueError(
            "keys= is not supported with promote_sam=True; promote explicit sample "
            "keys with promote_sample(sdata, keys)"
        )

    axes = ["obs", "var"] if which == "both" else [which]
    plans = [
        _plan_for_axis(sdata, ax, actions, keys, infer_types, dtypes, type_map, profile_kwargs)
        for ax in axes
    ]
    if promote_sam:
        sam_plan = promote_sample_suggested(
            sdata,
            infer_types=infer_types,
            dtypes=dtypes,
            type_map=type_map,
            prefix=prefix,
            dry_run=True,
            verbose=False,
        )
        # A sample-constant key is also a low-cardinality obs candidate, so keep
        # it out of the obs plan rather than materializing it on every cell.
        sam_keys = set(sam_plan["key"])
        plans = [
            p[~(p["key"].isin(sam_keys) & (p["which"] == "obs"))] for p in plans
        ]
        plans.append(sam_plan)

    plan = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame(
        columns=["which", "key", "dtype", "suggested_action"]
    )

    if verbose or dry_run:
        if plan.empty:
            print("promote_suggested: no keys matched — nothing to promote.")
        else:
            print(f"promote_suggested plan ({len(plan)} columns):")
            print(plan.to_string(index=False))

    if dry_run:
        return plan

    sam_rows = plan[plan["which"] == "sample"]
    if not sam_rows.empty:
        promote_sample(
            sdata,
            sam_rows["key"].tolist(),
            prefix=prefix,
            dtypes=dict(zip(sam_rows["key"], sam_rows["dtype"])),
        )

    for ax in axes:
        ax_plan = plan[plan["which"] == ax]
        ks = ax_plan["key"].tolist()
        if ks:
            promote(
                sdata,
                ks,
                which=ax,
                prefix=prefix,
                dtypes=dict(zip(ax_plan["key"], ax_plan["dtype"])),
            )
    return sdata
