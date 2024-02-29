import os
from dagster import Definitions


# from . import assets
from .assets.bronze_layer import *
from .assets.silver_layer import *
from .assets.gold_layer import *
from .resources.minio_io_manager import MinioIOManager
from .resources.mysql_io_manager import MysqlIOManager
from .resources.spark_io_manager import SparkIOManager



MYSQL_CONFIG = {
    "host": os.getenv('MYSQL_HOST'),
    "port": os.getenv('MYSQL_PORT'),
    "database": os.getenv('MYSQL_DATABASES'),
    "user": os.getenv('MYSQL_ROOT_USER'),
    "password": os.getenv('MYSQL_ROOT_PASSWORD')
}


MINIO_CONFIG = {
    "endpoint_url": os.getenv("MINIO_ENDPOINT"),
    "minio_access_key": os.getenv("MINIO_ACCESS_KEY"),
    "minio_secret_key": os.getenv("MINIO_SECRET_KEY"),
    "bucket": os.getenv("DATALAKE_BUCKET")
}

SPARK_CONFIG = {
    "spark_master": os.getenv("SPARK_MASTER_URL"),
    "endpoint_url": os.getenv("MINIO_ENDPOINT"),
    "minio_access_key": os.getenv("MINIO_ACCESS_KEY"),
    "minio_secret_key": os.getenv("MINIO_SECRET_KEY"),
    "bucket": os.getenv("DATALAKE_BUCKET")
}

resources = {
    "mysql_io_manager": MysqlIOManager(MYSQL_CONFIG),
    "minio_io_manager": MinioIOManager(MINIO_CONFIG),
    "spark_io_manager": SparkIOManager(SPARK_CONFIG),
} 

defs = Definitions(
    # assets=load_assets_from_modules([assets]),
    assets=[
        bronze_companies,
        bronze_trades,

        silver_cleaned_companies,
        silver_cleaned_trades,
        silver_dim_company,
        silver_dim_date,
        silver_fact_stock,

        gold_stock_price_change,
        gold_stock_gainer_daily,
        gold_stock_loser_daily,
    ],
    resources=resources
)