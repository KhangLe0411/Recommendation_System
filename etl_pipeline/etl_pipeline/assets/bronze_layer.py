from dagster import asset, Output
import pandas as pd
import boto3
import os

COMPUTE_KIND = "Python"
LAYER = "bronze"


def connect_minio():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        endpoint_url='http://' + os.getenv("MINIO_ENDPOINT")
    )
    try:
        return s3_client
    except Exception as e:
        raise e


def get_data_from_raw(name):
    client = connect_minio()
    response = client.get_object(Bucket=os.getenv(
        "DATALAKE_BUCKET"), Key=f'raw/{name}.csv')
    df = pd.read_csv(response.get("Body"), low_memory=False)
    return df


@asset(
    description="Load 'keywords' from raw layer using polars Dataframe and save to bronze layer",
    io_manager_key="minio_io_manager",
    key_prefix=["bronze", "keywords"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_keywords(context) -> Output[pd.DataFrame]:
    df = get_data_from_raw('keywords')
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "keywords",
            "row_count": df.shape[0],
            "column_count": df.shape[1]
        },
    )


@asset(
    description="Load 'movies' from raw layer using polars Dataframe and save to bronze layer",
    io_manager_key="minio_io_manager",
    key_prefix=["bronze", "movies"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_movies(context) -> Output[pd.DataFrame]:
    df = get_data_from_raw('movies_metadata')
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "movies",
            "row_count": df.shape[0],
            "column_count": df.shape[1]
        },
    )


@asset(
    description="Load 'credits' from raw layer using polars Dataframe and save to bronze layer",
    io_manager_key="minio_io_manager",
    key_prefix=["bronze", "credits"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_credits(context) -> Output[pd.DataFrame]:
    df = get_data_from_raw('credits')
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "credits",
            "row_count": df.shape[0],
            "column_count": df.shape[1]
        },
    )


@asset(
    description="Load 'ratings' from raw layer using polars Dataframe and save to bronze layer",
    io_manager_key="minio_io_manager",
    key_prefix=["bronze", "ratings"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_ratings(context) -> Output[pd.DataFrame]:
    df = get_data_from_raw('ratings')
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "ratings",
            "row_count": df.shape[0],
            "column_count": df.shape[1]
        },
    )


@asset(
    description="Load 'links' from raw layer using polars Dataframe and save to bronze layer",
    io_manager_key="minio_io_manager",
    key_prefix=["bronze", "links"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_links(context) -> Output[pd.DataFrame]:
    df = get_data_from_raw('links')
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "links",
            "row_count": df.shape[0],
            "column_count": df.shape[1]
        },
    )

# Sentiment

@asset(
    description="Load 'IMDB_Dataset' from raw layer using polars Dataframe and save to bronze layer",
    io_manager_key="minio_io_manager",
    key_prefix=["bronze", "review"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_review(context) -> Output[pd.DataFrame]:
    df = get_data_from_raw("IMDB_Dataset")
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "review",
            "row_count": df.shape[0],
            "column_count": df.shape[1]
        },
    )
