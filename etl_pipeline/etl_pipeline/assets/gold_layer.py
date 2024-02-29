from dagster import asset, AssetIn, Output
from ..resources.spark_io_manager import init_spark_session
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

COMPUTE_KIND = "PySpark"
LAYER = "gold"


@asset(
    description="Create table stock price change",
    ins={
        "silver_fact_stock": AssetIn(
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
    key_prefix=[LAYER, "price_change"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def gold_stock_price_change(context, silver_fact_stock: DataFrame, silver_dim_company: DataFrame, silver_dim_date: DataFrame) -> Output[DataFrame]:
    context.log.debug("Start creating price change table ...")

    with init_spark_session() as spark:
        # spark.sql("CREATE SCHEMA IF NOT EXISTS hive_prod.gold")
        spark_df = (
            silver_fact_stock
            .withColumn("day_price_change", col('close_price') - col('open_price'))
            .withColumn("percent_change", bround((col('day_price_change') / col('open_price')) * 100, 2))
            .join(silver_dim_date, 'dateKey', 'inner')
            .join(silver_dim_company, 'companyKey', 'inner')
            .select('companyKey', 'organShortName', 'full_date', 'open_price', 'close_price', 'Volume', 'day_price_change', 'percent_change')
        )

        return Output(
            value=spark_df,
            metadata={
                "table": "gold_stock_price_change",
                "row_count": spark_df.count(),
                "column_count": len(spark_df.columns),
                "columns": spark_df.columns,
            },
        )


@asset(
    description="Create table stock gainers",
    ins={
        "gold_stock_price_change": AssetIn(
            key_prefix=["gold", "price_change"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "stock_gainers"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def gold_stock_gainer_daily(context, gold_stock_price_change: DataFrame) -> Output[DataFrame]:
    context.log.debug("Start creating stock gainer daily table ...")

    spark_df = (
        gold_stock_price_change
        .withColumnRenamed("companyKey", "symbol")
        .filter(col('percent_change') > 0)
        .groupBy('symbol', 'organShortName', 'full_date')
        .agg(
            sum('open_price').alias('open'),
            sum('close_price').alias('close'),
            sum('day_price_change').alias('price_change'),
            sum('percent_change').alias('%_change'),
        )
        .orderBy(desc(max('full_date')), desc(max('percent_change')))
    )

    return Output(
        value=spark_df,
        metadata={
            "table": "gold_stock_gainer_daily",
            "row_count": spark_df.count(),
            "column_count": len(spark_df.columns),
            "columns": spark_df.columns,
        },
    )


@asset(
    description="Create table stock loser",
    ins={
        "gold_stock_price_change": AssetIn(
            key_prefix=["gold", "price_change"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "stock_losers"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def gold_stock_loser_daily(context, gold_stock_price_change: DataFrame) -> Output[DataFrame]:
    context.log.debug("Start creating stock loser daily table ...")

    spark_df = (
        gold_stock_price_change
        .withColumnRenamed("companyKey", "symbol")
        .filter(col('percent_change') <= 0)
        .groupBy('symbol', 'organShortName', 'full_date')
        .agg(
            sum('open_price').alias('open'),
            sum('close_price').alias('close'),
            sum('day_price_change').alias('price_change'),
            sum('percent_change').alias('%_change'),
        )
        .orderBy(desc(max('full_date')), asc(min('percent_change')))
    )

    return Output(
        value=spark_df,
        metadata={
            "table": "gold_stock_loser_daily",
            "row_count": spark_df.count(),
            "column_count": len(spark_df.columns),
            "columns": spark_df.columns,
        },
    )