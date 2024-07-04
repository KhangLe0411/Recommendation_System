from dagster import asset, AssetIn, Output
from ..resources.spark_io_manager import init_spark_session
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

COMPUTE_KIND = "PySpark"
LAYER = "gold"


@asset(
    description="Transform model",
    ins={
        "silver_merge_df": AssetIn(
            key_prefix=["silver", "final"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "transform_model"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def gold_transform_model(context, silver_merge_df: DataFrame) -> Output[DataFrame]:
    context.log.debug("Start transform model ...")
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("vecs", ArrayType(DoubleType()), True)
    ])
    spark_df = silver_merge_df.select('id', 'comb')
    pipeline_mdl = PipelineModel.load("s3a://lakehouse/model/" + 'pipeline_model')
    new_df = pipeline_mdl.transform(spark_df)
    all_movies_vecs = new_df.select('id', 'word_vec').rdd.map(lambda x: (x[0], x[1])).collect()
    data = [(id, [float(x) for x in vec]) for id, vec in all_movies_vecs]
    with init_spark_session() as spark:
        all_movies_df = spark.createDataFrame(data, schema)

    return Output(
        value=all_movies_df,
        metadata={
            "table": "gold_transform_model",
            "row_count": all_movies_df.count(),
            "column_count": len(all_movies_df.columns),
            "columns": all_movies_df.columns,
        },
    )
