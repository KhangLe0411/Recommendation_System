from typing import Any
from dagster import IOManager, InputContext, OutputContext
from contextlib import contextmanager
from sqlalchemy import create_engine
import polars as pl

def connect_mysql(config) -> str:
    # Ex: mysql://user:pass@host:port/db_name
    conn = (
        f"mysql://{config['user']}:{config['password']}"
        + f"@{config['host']}:{config['port']}"
        + f"/{config['database']}"
    )
    return conn

class MysqlIOManager(IOManager):
    def __init__(self, config):
        self._config = config

    def handle_output(self, context: OutputContext, obj: pl.DataFrame):
        pass

    def load_input(self, context: InputContext):
        pass

    def extract_data(self, sql: str):
        """
            Extract data from MySQL
        """
        conn = connect_mysql(self._config)
        df = pl.read_database(query=sql, connection_uri=conn)
        return df