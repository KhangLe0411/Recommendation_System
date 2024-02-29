from dagster import asset, AssetIn, Output
import polars as pl

from pyspark.sql import DataFrame

from ..resources.spark_io_manager import init_spark_session
from pyspark.sql.functions import *
from datetime import timedelta
from pyspark.sql.types import *

COMPUTE_KIND = "PySpark"
LAYER = "silver"

@asset(
    description="Cleaning companies table",
    ins={
        "bronze_companies": AssetIn(
            key_prefix=["bronze", "company"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "company"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_companies(context, bronze_companies: pl.DataFrame) -> Output[DataFrame]:
    """
        Load companies table from bronze layer in MinIO, into a Spark dataframe, then clean data
    """

    context.log.debug("Start cleaning companies table")

    with init_spark_session() as spark:
        # Convert bronze_trade from polars DataFrame to Spark DataFrame
        pandas_df = bronze_companies.to_pandas()
        context.log.debug(
            f"Converted to pandas DataFrame with shape: {pandas_df.shape}"
        )

        spark_df = spark.createDataFrame(pandas_df)

        context.log.info("Got Spark DataFrame")
        # spark.sql("CREATE SCHEMA IF NOT EXISTS hive_prod.silver")

        return Output(
            value=spark_df,
            metadata={
                "table": "silver_cleaned_companies",
                "row_count": spark_df.count(),
                "column_count": len(spark_df.columns),
                "columns": spark_df.columns,
            },
        )

@asset(
    description="Cleaning trades table",
    ins={
        "bronze_trades": AssetIn(
            key_prefix=["bronze", "trade"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "trade"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_trades(context, bronze_trades: pl.DataFrame) -> Output[DataFrame]:
    """
        Load trade table from bronze layer in MinIO, into a Spark dataframe, then clean data
    """

    context.log.debug("Start cleaning trades table")

    with init_spark_session() as spark:
        # Convert bronze_trade from polars DataFrame to Spark DataFrame
        pandas_df = bronze_trades.to_pandas()
        context.log.debug(
            f"Converted to pandas DataFrame with shape: {pandas_df.shape}"
        )

        spark_df = spark.createDataFrame(pandas_df)

        context.log.info("Got Spark DataFrame")
        spark_df = spark_df \
            .withColumnRenamed("open", "open_price") \
            .withColumnRenamed("high", "high_price") \
            .withColumnRenamed("low", "low_price") \
            .withColumnRenamed("close", "close_price") \
            .drop('ID') \
            .dropDuplicates(['symbol', 'tradingDate'])

        return Output(
            value=spark_df,
            metadata={
                "table": "silver_cleaned_trades",
                "row_count": spark_df.count(),
                "column_count": len(spark_df.columns),
                "columns": spark_df.columns,
            },
        )
    
@asset(
    description="Create dim company",
    ins={
        "silver_cleaned_companies": AssetIn(
            key_prefix=["silver", "company"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "company"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_dim_company(context, silver_cleaned_companies: DataFrame) -> Output[DataFrame]:
    spark_df = silver_cleaned_companies
    context.log.debug("Start creating dim company table")

    spark_df = spark_df \
        .withColumnRenamed("symbol", "companyKey") \
        .select("companyKey", "comGroupCode", "organName", "organShortName", "organTypeCode", "icbName", "sector", "industry", "group", "subGroup")

    return Output(
        value=spark_df,
        metadata={
            "table": "silver_dim_company",
            "row_count": spark_df.count(),
            "column_count": len(spark_df.columns),
            "columns": spark_df.columns,
        },
    )

@asset(
    description="Create dim date",
    ins={
        "silver_cleaned_trades": AssetIn(
            key_prefix=["silver", "trade"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "trade"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_dim_date(context, silver_cleaned_trades: DataFrame) -> Output[DataFrame]:
    spark_df = silver_cleaned_trades
    context.log.debug("Start creating dim date ")
    
    start_date = spark_df.select(min("tradingDate")).first()[0]
    end_date = spark_df.select(max("tradingDate")).first()[0]

    with init_spark_session() as spark:
        date_range = spark.sparkContext.parallelize([(start_date + timedelta(days=x)) for x in range((end_date - start_date).days + 1)])
        date_df = date_range.map(lambda x: (x,)).toDF(['date'])

        date_df = date_df.withColumn("dateKey", year(col("date"))*10000 + month(col("date"))*100 + dayofmonth(col("date"))) \
            .withColumn("year", year(col("date"))) \
            .withColumn("quarter", quarter(col("date"))) \
            .withColumn("month", month(col("date"))) \
            .withColumn("week", weekofyear(col("date"))) \
            .withColumn("day", dayofmonth(col("date"))) \
            .withColumn("day_of_year", dayofyear(col("date"))) \
            .withColumn("day_name_of_week", date_format(col("date"), "EEEE")) \
            .withColumn("month_name_of_week", date_format(col("date"), "MMMM")) \
            .withColumnRenamed("date", "full_date") \
            .selectExpr(['dateKey', 'full_date', 'year', 'quarter', 'month', 'week', 'day', 'day_of_year', 'day_name_of_week', 'month_name_of_week'])

        return Output(
            value=date_df,
            metadata={
                "table": "silver_dim_date",
                "row_count": date_df.count(),
                "column_count": len(date_df.columns),
                "columns": date_df.columns,
            },
        )

@asset(
    description="Create fact table",
    ins={
        "silver_cleaned_trades": AssetIn(
            key_prefix=["silver", "trade"]
        ),
        "silver_dim_company": AssetIn(
            key_prefix=["silver", "company"]
        ),
        "silver_dim_date": AssetIn(
            key_prefix=["silver", "trade"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "trade"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_fact_stock(context, silver_cleaned_trades: DataFrame, silver_dim_company: DataFrame, silver_dim_date: DataFrame) -> Output[DataFrame]:
    context.log.debug("Start creating fact table ...")
    
    fact_table = (
        silver_cleaned_trades
        .join(silver_dim_company, silver_cleaned_trades['symbol'] == silver_dim_company['companyKey'], 'inner')
        .join(silver_dim_date, silver_cleaned_trades['tradingDate'] == silver_dim_date['full_date'], 'inner')
        .select('dateKey', 'companyKey', 'open_price', 'high_price', 'low_price', 'close_price', 'Volume')
    )
    
    return Output(
        value=fact_table,
        metadata={
            "table": "silver_fact_stock",
            "row_count": fact_table.count(),
            "column_count": len(fact_table.columns),
            "columns": fact_table.columns,
        },
    )