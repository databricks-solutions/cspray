import os

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark_collect():
    """Reuse a cluster session if one exists. Otherwise start local Spark with Delta.

    ``configure_spark_with_delta_pip`` sets ``spark.jars.packages`` so Ivy can
    fetch the Delta JARs. Where Ivy downloads from is the environment (Maven
    Central by default; set ``spark.jars.repositories`` / ``PYSPARK_SUBMIT_ARGS``
    if you need a proxy). Databricks already has Delta — do not download there.
    """
    existing = SparkSession.getActiveSession()
    if existing is not None:
        return existing
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return SparkSession.builder.getOrCreate()

    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder
        .appName("CsprayWorkflow")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()
