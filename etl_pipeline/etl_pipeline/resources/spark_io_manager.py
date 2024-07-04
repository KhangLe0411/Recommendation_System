from dagster import IOManager, InputContext, OutputContext
from pyspark.sql import SparkSession, DataFrame

from contextlib import contextmanager
import os


@contextmanager
def init_spark_session():
    try:
        spark = (
            SparkSession.builder.master("spark://spark-master:7077")
            .appName("Spark IO Manager")
            .config("spark.jars", "/usr/local/spark/jars/hadoop-aws-3.3.2.jar,/usr/local/spark/jars/hadoop-common-3.3.2.jar,/usr/local/spark/jars/aws-java-sdk-1.12.367.jar,/usr/local/spark/jars/s3-2.18.41.jar,/usr/local/spark/jars/aws-java-sdk-bundle-1.11.1026.jar,/usr/local/spark/jars/iceberg-spark-runtime-3.3_2.12-1.5.2.jar")
            .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
            .config("spark.sql.catalog.spark_catalog.type", "hive")
            .config(f"spark.sql.catalog.hive_prod", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.hive_prod.type", "hive")
            .config(f"spark.sql.catalog.hive_prod.uri", "thrift://hive-metastore:9083")
            .config("spark.sql.warehouse.dir", f"s3a://{os.getenv('DATALAKE_BUCKET')}/")
            .config("spark.hadoop.fs.s3a.endpoint", f"http://{os.getenv('MINIO_ENDPOINT')}")
            .config("spark.hadoop.fs.s3a.access.key", f"{os.getenv('MINIO_ACCESS_KEY')}")
            .config("spark.hadoop.fs.s3a.secret.key", f"{os.getenv('MINIO_SECRET_KEY')}")
            .config(
                "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
            )
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.connection.ssl.enabled", "false")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "2g")
            .config("spark.rpc.message.maxSize", "800")
            .config("spark.sql.execution.pythonUDF.arrow.enabled", "true")
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .enableHiveSupport()
            .getOrCreate()
        )
        yield spark
    except Exception as e:
        raise e


class SparkIOManager(IOManager):
    def __init__(self, config):
        self._config = config

    def handle_output(self, context: OutputContext, obj: DataFrame):
        """
            Write output to MinIO as parquet file
        """
        context.log.debug("(Spark handle_output) Writing output to MinIO ...")

        layer, _, table = context.asset_key.path
        table_name = str(table.replace(f"{layer}_", ""))
        mode = (context.metadata or {}).get("mode", "overwrite")
        context.log.debug(
            f"(Spark handle_output) Layer: {layer} - table: {table_name}")
        context.log.debug(f"mode: {mode}")

        try:
            with init_spark_session() as spark:
                spark.sql(f"CREATE SCHEMA IF NOT EXISTS hive_prod.{layer}")

            obj.write.format("iceberg").mode(f"{mode}").saveAsTable(
                f"hive_prod.{layer}.{table_name}")
            # context.log.debug(f"Table: {obj.show(3)}")
            context.log.debug(f"Saved {table_name} to {layer} layer")
        except Exception as e:
            raise Exception(
                f"(Spark handle_output) Error while writing output: {e}")

    def load_input(self, context: InputContext) -> DataFrame:
        """
            Load input from minio from parquet file to spark dataframe
        """
        # E.g context.asset_key.path: ['silver', 'stock', 'stocks']
        layer, _, table = context.asset_key.path
        table_name = table.replace(f"{layer}_", "")
        context.log.debug(
            f'Loading input from {layer} layer - table {table_name}...')
        try:
            with init_spark_session() as spark:
                df = spark.read.format("iceberg").load(
                    f"hive_prod.{layer}.{table_name}")
                context.log.debug(
                    f"Loaded {df.count()} rows from table {table_name}")
                return df
        except Exception as e:
            raise Exception(f"Error while loading input: {e}")
