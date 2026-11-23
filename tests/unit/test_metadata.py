import json
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from cspray.data import SAM_REGISTRY_COLUMNS, SprayData
import cspray as cs

def test_metadata_dummy_roundtrip(dummy_h5ad_pair, spark_collect):
    """End-to-end metadata check on two hand-built h5ad files with known values:
    pick-up, exact per-row assignment, coverage/action classification, per-file
    localisation, and promotion. Uses metadata_variant=False for OSS portability.

    max_categorical=3 makes the id-like/categorical split deterministic at this
    tiny scale (the default present>50 id-like rule cannot fire on 6 rows).
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )

    # 1. sam exists immediately at one row per input file -------------------
    sam = sdata.sam.toPandas()
    assert len(sam) == 2
    assert set(sam.columns) == set(SAM_REGISTRY_COLUMNS)
    assert set(sam['file_path']) == set(dummy_h5ad_pair)

    # 2. every cell carries metadata, keyed to the right barcode ------------
    obs = sdata.obs.select('cell_barcode', 'obs_data').toPandas()
    assert set(obs['cell_barcode']) == {'cA0','cA1','cA2','cB0','cB1','cB2'}
    parsed = {r.cell_barcode: json.loads(r.obs_data) for r in obs.itertuples()}
    assert parsed['cA0']['cell_type'] == 'T'
    assert parsed['cA0']['n_genes'] == 10
    assert parsed['cB2']['cell_type'] == 'B'
    assert parsed['cB2']['batch'] == 'b2'
    assert 'batch' not in parsed['cA0']       # file A never had it
    assert 'CellType' not in parsed['cB0']    # file B never had it

    # 3. profile: pick-up + coverage + classification -----------------------
    r = cs.md.profile(sdata, which='obs', max_categorical=3, verbose=False).set_index('key')
    assert set(r.index) == {
        'cell_type','tissue','donor_id','organism','n_genes','qc_score',
        'CellType','batch','mixed'}
    assert r.loc['cell_type','coverage_pct'] == 100.0 and r.loc['cell_type','present'] == 6
    assert r.loc['batch','coverage_pct'] == 50.0 and r.loc['batch','tag'] == 'partial'
    assert r.loc['CellType','coverage_pct'] == 50.0

    assert r.loc['organism','suggested_action'] == 'drop_constant'
    assert r.loc['cell_type','suggested_action'] == 'promote'
    assert r.loc['donor_id','suggested_action'] == 'keep_in_variant'   # nd=6 > 3
    assert bool(r.loc['cell_type','possible_alias']) and bool(r.loc['CellType','possible_alias'])
    assert bool(r.loc['mixed','type_conflict'])

    # 4. per-file matrix -----------------------------------------------------
    _, matrix = cs.md.profile(sdata, which='obs', per_file=True, verbose=False)
    assert matrix.shape[1] == 2
    assert set(matrix.loc['batch']) == {0.0, 100.0}
    assert (matrix.loc['cell_type'] == 100.0).all()

    # 5. promote (string + typed) -------------------------------------------
    cs.md.promote(sdata, 'cell_type', which='obs')
    cs.md.promote(sdata, 'n_genes', which='obs', dtypes='int')
    prom = sdata.obs.select('cell_barcode','cell_type','n_genes').toPandas().set_index('cell_barcode')
    assert prom.loc['cA0','cell_type'] == 'T'
    assert int(prom.loc['cA1','n_genes']) == 20
    assert int(prom.loc['cB2','n_genes']) == 35

    # 6. var side, briefly ---------------------------------------------------
    vr = cs.md.profile(sdata, which='var', max_categorical=3, verbose=False).set_index('key')
    assert {'feature_type','highly_variable'}.issubset(set(vr.index))
    assert vr.loc['feature_type','suggested_action'] == 'drop_constant'


def test_sample_metadata_promotion_and_qc_order(dummy_h5ad_pair, spark_collect):
    """Sample metadata and QC independently enrich the read-time sam table."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)

    plan = cs.md.promote_sample_suggested(sdata, dry_run=True, verbose=False)
    assert set(plan.columns) == {'which', 'key', 'dtype', 'suggested_action'}
    assert set(plan['which']) == {'sample'}
    assert {'tissue', 'organism'}.issubset(set(plan['key']))
    assert 'cell_type' not in set(plan['key'])
    assert 'donor_id' not in set(plan['key'])

    # If a sample-level key was previously promoted on obs, moving it to sam
    # removes only that duplicated top-level column (the payload remains).
    cs.md.promote(sdata, 'tissue', which='obs')
    assert 'tissue' in sdata.obs.columns
    cs.md.promote_sample_suggested(sdata, verbose=False)
    sam = sdata.sam.select('file_path', 'tissue', 'organism').toPandas().set_index('file_path')
    assert sam.loc[dummy_h5ad_pair[0], 'tissue'] == 'lung'
    assert sam.loc[dummy_h5ad_pair[1], 'tissue'] == 'blood'
    assert set(sam['organism']) == {'human'}
    assert 'tissue' not in sdata.obs.columns
    assert 'tissue' in json.loads(
        sdata.obs.select('obs_data').limit(1).toPandas()['obs_data'].iloc[0]
    )

    cs.pp.calculate_qc_metrics(sdata)
    assert {'tissue', 'organism', 'n_cells', 'total_counts',
            'mean_genes_per_cell'}.issubset(set(sdata.sam.columns))

    external = spark_collect.createDataFrame([
        (dummy_h5ad_pair[0], 'study-a'),
        (dummy_h5ad_pair[1], 'study-b'),
    ], ['file_path', 'study'])
    cs.md.add_sample_metadata(sdata, external)
    assert set(sdata.sam.select('study').toPandas()['study']) == {'study-a', 'study-b'}


def test_qc_before_sample_metadata_promotion(dummy_h5ad_pair, spark_collect):
    """Promoting after QC preserves previously calculated sample metrics."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    cs.pp.calculate_qc_metrics(sdata)
    cs.md.promote_sample(sdata, ['tissue'], dtypes='string')

    assert {'tissue', 'n_cells', 'total_counts',
            'mean_genes_per_cell'}.issubset(set(sdata.sam.columns))
    assert set(sdata.sam.select('tissue').toPandas()['tissue']) == {'lung', 'blood'}


def test_promote_suggested_with_promote_sam(dummy_h5ad_pair, spark_collect):
    """promote_sam=True routes sample-constant keys to sam in the same call, and
    keeps them out of the obs plan so nothing is materialized per cell."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)

    plan = cs.md.promote_suggested(
        sdata, which='both', profile_kwargs={'max_categorical': 3},
        promote_sam=True, dry_run=True, verbose=False,
    )
    by_target = {w: set(g['key']) for w, g in plan.groupby('which')}
    assert {'tissue', 'organism'}.issubset(by_target['sample'])
    assert 'tissue' not in by_target['obs']       # excluded in favour of sam
    assert 'cell_type' in by_target['obs']        # varies per cell, stays on obs
    assert 'highly_variable' in by_target['var']

    cs.md.promote_suggested(
        sdata, which='both', profile_kwargs={'max_categorical': 3},
        promote_sam=True, verbose=False,
    )
    assert 'cell_type' in sdata.obs.columns
    assert 'tissue' not in sdata.obs.columns
    sam = sdata.sam.select('file_path', 'tissue').toPandas().set_index('file_path')
    assert sam.loc[dummy_h5ad_pair[0], 'tissue'] == 'lung'
    assert sam.loc[dummy_h5ad_pair[1], 'tissue'] == 'blood'

    with pytest.raises(ValueError):
        cs.md.promote_suggested(sdata, which='obs', keys=['tissue'], promote_sam=True)


def test_promote_json_path_helpers():
    """JSONPath quoting and Spark identifier sanitising used by promotion."""
    from cspray.metadata import _json_path, _safe_column_name

    assert _json_path("orig.ident") == "$['orig.ident']"
    assert _json_path("cell type") == "$['cell type']"
    assert _json_path("a'b") == '$["a\'b"]'
    assert _json_path('a"b') == '$[\'a"b\']'
    assert _json_path("""a'"b""") is None
    assert _json_path("") is None
    assert _safe_column_name("orig.ident") == "orig_ident"
    assert _safe_column_name("cell type") == "cell_type"
    assert _safe_column_name("2bad") == "_2bad"


def test_promote_keys_with_dots(spark_collect, tmp_path):
    """Regression: payload keys with '.' (e.g. Seurat orig.ident) must extract
    via bracket JSONPath and land as Spark-safe column names, including sam.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    adata = ad.AnnData(
        X=csr_matrix(np.arange(8, dtype="float32").reshape(2, 4)),
        obs=pd.DataFrame({
            "orig.ident": ["s1", "s1"],
            "cell.type": pd.Categorical(["T", "B"]),
        }, index=["c0", "c1"]),
        var=pd.DataFrame({
            "gene.biotype": ["protein_coding", "lncRNA", "protein_coding", "lncRNA"],
        }, index=["g0", "g1", "g2", "g3"]),
    )
    path = str(tmp_path / "dots.h5ad")
    adata.write(path)

    sdata = SprayData.from_h5ads(
        spark_collect, path=[path], from_raw=False, mode="delta",
        obs_metadata_columns="all", var_metadata_columns="all",
        metadata_variant=False, force_partitioning=1,
    )
    sdata.set_intermediary_persistance(False)

    plan = cs.md.promote_suggested(
        sdata, which="both", promote_sam=True, dry_run=True, verbose=False,
        profile_kwargs={"max_categorical": 3},
    )
    by_target = {w: set(g["key"]) for w, g in plan.groupby("which")}
    assert "orig.ident" in by_target["sample"]
    assert "cell.type" in by_target["obs"]
    assert "gene.biotype" in by_target["var"]

    cs.md.promote_suggested(
        sdata, which="both", promote_sam=True, verbose=False,
        profile_kwargs={"max_categorical": 3},
    )

    assert "orig_ident" in sdata.sam.columns
    assert "orig.ident" not in sdata.sam.columns
    assert sdata.sam.select("orig_ident").toPandas()["orig_ident"].iloc[0] == "s1"

    assert "cell_type" in sdata.obs.columns
    assert "cell.type" not in sdata.obs.columns
    obs = sdata.obs.select("cell_barcode", "cell_type").toPandas().set_index("cell_barcode")
    assert obs.loc["c0", "cell_type"] == "T"
    assert obs.loc["c1", "cell_type"] == "B"
    assert "orig.ident" in json.loads(
        sdata.obs.select("obs_data").limit(1).toPandas()["obs_data"].iloc[0]
    )

    assert "gene_biotype" in sdata.var.columns
    assert "gene.biotype" not in sdata.var.columns
    var = sdata.var.select("gene_id", "gene_biotype").toPandas().set_index("gene_id")
    assert var.loc["g0", "gene_biotype"] == "protein_coding"
    assert var.loc["g1", "gene_biotype"] == "lncRNA"


def test_metadata_promote_suggested(dummy_h5ad_pair, spark_collect):
    """promote_suggested: dry_run returns an editable plan without mutating, and
    applying it materializes the recommended keys with inferred Spark types.

    max_categorical=3 (via profile_kwargs) makes the classification deterministic
    at this tiny scale, so cell_type and tissue (obs) and highly_variable (var)
    are 'promote'; organism is drop_constant and the unique/mixed keys
    keep_in_variant. tissue is sample-constant, so it is only an obs candidate
    while promote_sam is off (see test_promote_suggested_with_promote_sam).
    """
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )

    # --- dry_run: plan only, no mutation -----------------------------------
    plan = cs.md.promote_suggested(
        sdata, which='obs', profile_kwargs={'max_categorical': 3},
        dry_run=True, verbose=False,
    )
    assert isinstance(plan, pd.DataFrame)
    assert set(plan.columns) == {'which', 'key', 'dtype', 'suggested_action'}
    plan_keys = set(plan['key'])
    assert 'cell_type' in plan_keys        # promote
    assert 'tissue' in plan_keys           # promote (sample-constant, but promote_sam off)
    assert 'organism' not in plan_keys     # drop_constant
    assert 'donor_id' not in plan_keys     # keep_in_variant (nd=6 > 3)
    assert plan.set_index('key').loc['cell_type', 'dtype'] == 'string'
    assert 'cell_type' not in sdata.obs.columns   # dry_run must not mutate

    # --- apply across both axes, typed -------------------------------------
    cs.md.promote_suggested(
        sdata, which='both', profile_kwargs={'max_categorical': 3}, verbose=False,
    )
    assert 'cell_type' in sdata.obs.columns          # obs promote
    assert 'highly_variable' in sdata.var.columns    # var promote
    # inferred types: str -> string, bool -> boolean
    assert sdata.obs.schema['cell_type'].dataType.simpleString() == 'string'
    assert sdata.var.schema['highly_variable'].dataType.simpleString() == 'boolean'
    obs = sdata.obs.select('cell_barcode', 'cell_type').toPandas().set_index('cell_barcode')
    assert obs.loc['cA0', 'cell_type'] == 'T'

    # --- guard: keys override not allowed with which='both' ----------------
    with pytest.raises(ValueError):
        cs.md.promote_suggested(sdata, which='both', keys=['cell_type'])


def test_promote_applies_dry_run_plan(dummy_h5ad_pair, spark_collect):
    """promote(sdata, plan=) is the apply step for a dry_run plan."""
    sdata = SprayData.from_h5ads(
        spark_collect, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    plan = cs.md.promote_suggested(
        sdata, which='both', promote_sam=True, dry_run=True, verbose=False,
        profile_kwargs={'max_categorical': 3},
    )
    plan = plan[plan['key'].isin(['cell_type', 'tissue', 'highly_variable'])]
    cs.md.promote(sdata, plan=plan)

    assert 'cell_type' in sdata.obs.columns
    assert 'tissue' not in sdata.obs.columns
    assert 'tissue' in sdata.sam.columns
    assert 'highly_variable' in sdata.var.columns
    obs = sdata.obs.select('cell_barcode', 'cell_type').toPandas().set_index('cell_barcode')
    assert obs.loc['cA0', 'cell_type'] == 'T'


def test_promote_dest_from_suggested_and_plan(dummy_h5ad_pair, spark_collect, tmp_path):
    """Dest rewrite writes obs/var/sam only; bronze and X stay put."""
    spark = spark_collect
    bronze = str(tmp_path / 'bronze')
    dest_suggested = str(tmp_path / 'dest_suggested')
    dest_plan = str(tmp_path / 'dest_plan')

    sdata = SprayData.from_h5ads(
        spark, path=dummy_h5ad_pair, from_raw=False, mode='delta',
        obs_metadata_columns='all', var_metadata_columns='all',
        metadata_variant=False, force_partitioning=2,
    )
    sdata.set_intermediary_persistance(False)
    plan = cs.md.promote_suggested(
        sdata, which='both', promote_sam=True, dry_run=True, verbose=False,
        profile_kwargs={'max_categorical': 3},
    )
    plan = plan[plan['key'].isin(['cell_type', 'tissue', 'highly_variable'])]
    sdata.to_tables(spark, table_base=bronze, join_char='/')

    cs.md.promote(
        spark, bronze, dest_suggested, suggested=True, join_char='/',
        mode='delta', profile_kwargs={'max_categorical': 3},
    )
    cs.md.promote(
        spark, bronze, dest_plan, plan=plan, join_char='/', mode='delta',
    )

    bronze_loaded = SprayData.from_tables(
        spark, bronze, join_char='/', mode='delta',
    )
    assert 'cell_type' not in bronze_loaded.obs.columns
    assert bronze_loaded.X is not None

    for dest in (dest_suggested, dest_plan):
        loaded = SprayData.from_tables(spark, dest, join_char='/', mode='delta')
        assert loaded.X is None
        assert not (Path(dest) / 'X').exists()
        assert 'cell_type' in loaded.obs.columns
        assert 'tissue' not in loaded.obs.columns
        assert 'tissue' in loaded.sam.columns
        assert 'highly_variable' in loaded.var.columns
        obs = loaded.obs.select('cell_barcode', 'cell_type').toPandas().set_index(
            'cell_barcode'
        )
        assert obs.loc['cA0', 'cell_type'] == 'T'
        sam = loaded.sam.select('file_path', 'tissue').toPandas().set_index('file_path')
        assert sam.loc[dummy_h5ad_pair[0], 'tissue'] == 'lung'
        assert sam.loc[dummy_h5ad_pair[1], 'tissue'] == 'blood'


def test_promote_dest_guards(spark_collect, tmp_path):
    bronze = str(tmp_path / 'bronze')
    dest = str(tmp_path / 'dest')
    with pytest.raises(ValueError, match='plan='):
        cs.md.promote(spark_collect, bronze, dest, join_char='/', mode='delta')
    with pytest.raises(ValueError, match='differ'):
        cs.md.promote(
            spark_collect, bronze, bronze, suggested=True,
            join_char='/', mode='delta',
        )
    with pytest.raises(ValueError, match='not both'):
        cs.md.promote(
            spark_collect, bronze, dest, plan=pd.DataFrame(),
            suggested=True, join_char='/', mode='delta',
        )
