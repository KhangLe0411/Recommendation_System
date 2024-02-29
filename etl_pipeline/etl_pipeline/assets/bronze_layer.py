from dagster import asset, Output
import polars as pl

COMPUTE_KIND = "SQL"
LAYER = "bronze"

@asset(
        description="Load table 'companies' from MySQL using polars Dataframe and save to MinIO",
        io_manager_key="minio_io_manager",
        required_resource_keys={"mysql_io_manager"},
        key_prefix=["bronze", "company"],
        compute_kind=COMPUTE_KIND,
        group_name=LAYER
)
def bronze_companies(context) -> Output[pl.DataFrame]:
    query = "SELECT * FROM companies;"
    df = context.resources.mysql_io_manager.extract_data(query)
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "companies",
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "columns": df.columns
        },
    )

@asset(
    description="Load table 'Trades' from MySQL using polars Dataframe and save to MinIO",
    io_manager_key="minio_io_manager",
    required_resource_keys={"mysql_io_manager"},
    key_prefix=["bronze", "trade"],
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def bronze_trades(context) -> Output[pl.DataFrame]:
    query = "SELECT * FROM Trades;"
    df = context.resources.mysql_io_manager.extract_data(query)
    context.log.info(f"Table extracted with shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "table": "trades",
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "columns": df.columns
        },
    )