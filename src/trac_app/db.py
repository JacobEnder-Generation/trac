from __future__ import annotations

from typing import Optional

import pandas as pd
import pyodbc

from .env import get_env_value


def _build_connection_string() -> str:
    server = get_env_value("SQL_SERVER")
    database = get_env_value("SQL_DATABASE")
    username = get_env_value("SQL_USERNAME")
    password = get_env_value("SQL_PASSWORD")

    missing = [
        name
        for name, value in {
            "SQL_SERVER": server,
            "SQL_DATABASE": database,
            "SQL_USERNAME": username,
            "SQL_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise ConnectionError(
            f"Missing database configuration values: {', '.join(sorted(missing))}. "
            "Set them as environment variables or in assets/.env."
        )

    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )


class DatabaseConnection:
    def __init__(self) -> None:
        self.conn: Optional[pyodbc.Connection] = None

    def connect(self) -> pyodbc.Connection:
        if self.conn is not None:
            return self.conn
        try:
            conn_str = _build_connection_string()
            self.conn = pyodbc.connect(conn_str)
        except Exception as exc:  # pragma: no cover - shown via Streamlit
            raise ConnectionError(f"Database connection error: {exc}") from exc
        return self.conn

    def disconnect(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> pyodbc.Connection:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class DataFetcher:
    @staticmethod
    def get_available_tickers(conn: pyodbc.Connection) -> list[str]:
        query = """
        SELECT DISTINCT tr.ticker_region
        FROM sym_v1.sym_ticker_region tr
        JOIN ff_V3.FF_BASIC_AF AF ON tr.fsym_id = af.fsym_id
        ORDER BY tr.ticker_region
        """
        return pd.read_sql(query, conn)["ticker_region"].tolist()

    @staticmethod
    def get_fundamentals_data(conn: pyodbc.Connection, ticker: str) -> pd.DataFrame:
        query = """
        SELECT 
            tr.ticker_region,
            af.DATE,
            af.ff_com_shs_out,
            af.ff_com_eq
        FROM ff_V3.FF_BASIC_QF AF
        JOIN sym_v1.sym_ticker_region tr ON tr.fsym_id = af.fsym_id
        WHERE tr.ticker_region = ?
        AND af.DATE >= DATEADD(YEAR, -10, GETDATE())
        ORDER BY af.DATE DESC
        """
        df = pd.read_sql(query, conn, params=[ticker])
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    @staticmethod
    def get_daily_data(conn: pyodbc.Connection, ticker: str) -> pd.DataFrame:
        query = """
        SELECT 
            tr.ticker_region,
            bp.p_date,
            bp.p_price_open as [open],
            bp.p_price_high as [high],
            bp.p_price_low as [low],
            bp.p_price as [close]
        FROM FP_V2.fp_basic_prices bp
        JOIN sym_v1.sym_ticker_region tr ON tr.fsym_id = bp.fsym_id
        WHERE tr.ticker_region = ?
        AND bp.p_date >= DATEADD(YEAR, -10, GETDATE())
        ORDER BY p_date DESC
        """
        df = pd.read_sql(query, conn, params=[ticker])
        df["p_date"] = pd.to_datetime(df["p_date"])
        return df

    @staticmethod
    def get_forecast_data(conn: pyodbc.Connection, ticker: str) -> pd.DataFrame:
        query = """
        SELECT 
            tr.ticker_region,
            fe.fe_fp_end as date,
            fe.fe_mean as bps_forecast
        FROM fe_v4.fe_advanced_conh_af fe
        JOIN sym_v1.sym_ticker_region tr ON tr.fsym_id = fe.fsym_id
        WHERE tr.ticker_region = ?
        AND fe.fe_item = 'BPS'
        AND fe.fe_fp_end > GETDATE()
        ORDER BY fe.fe_fp_end ASC
        """
        df = pd.read_sql(query, conn, params=[ticker])
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def get_split_history(conn: pyodbc.Connection, ticker: str) -> pd.DataFrame:
        query = """
        SELECT 
            tr.ticker_region,
            sp.p_split_date as split_date,
            sp.p_split_factor as split_factor
        FROM fp_v2.fp_basic_splits sp
        JOIN sym_v1.sym_ticker_region tr ON tr.fsym_id = sp.fsym_id
        WHERE tr.ticker_region = ?
        AND sp.p_split_date >= DATEADD(YEAR, -10, GETDATE())
        ORDER BY p_split_date DESC
        """
        df = pd.read_sql(query, conn, params=[ticker])
        df["split_date"] = pd.to_datetime(df["split_date"])
        return df

    @staticmethod
    def get_etf_tickers(conn: pyodbc.Connection) -> list[str]:
        query = """
        SELECT DISTINCT tr.ticker_region
        FROM sym_v1.sym_ticker_region tr
        JOIN fe_v4.fe_advanced_conh_af af ON tr.fsym_id = af.fsym_id
        WHERE af.fe_item = 'BPS'
        AND af.fe_mean IS NOT NULL
        ORDER BY tr.ticker_region
        """
        return pd.read_sql(query, conn)["ticker_region"].tolist()

    @staticmethod
    def get_etf_bvps_data(conn: pyodbc.Connection, ticker: str) -> pd.DataFrame:
        query = """
        SELECT 
            tr.ticker_region,
            af.fe_fp_end as DATE,
            af.fe_mean as bvps
        FROM fe_v4.fe_advanced_conh_af af
        JOIN sym_v1.sym_ticker_region tr ON tr.fsym_id = af.fsym_id
        WHERE tr.ticker_region = ?
        AND af.fe_item = 'BPS'
        AND af.fe_mean IS NOT NULL
        AND af.fe_fp_end >= DATEADD(YEAR, -5, GETDATE())
        ORDER BY af.fe_fp_end DESC
        """
        df = pd.read_sql(query, conn, params=[ticker])
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    @staticmethod
    def get_etf_daily_data(conn: pyodbc.Connection, ticker: str) -> pd.DataFrame:
        query = """
        SELECT 
            tr.ticker_region,
            bp.p_date,
            bp.p_price_open as [open],
            bp.p_price_high as [high],
            bp.p_price_low as [low],
            bp.p_price as [close]
        FROM FP_V2.fp_basic_prices bp
        JOIN sym_v1.sym_ticker_region tr ON tr.fsym_id = bp.fsym_id
        WHERE tr.ticker_region = ?
        AND bp.p_date >= DATEADD(YEAR, -5, GETDATE())
        ORDER BY p_date DESC
        """
        df = pd.read_sql(query, conn, params=[ticker])
        df["p_date"] = pd.to_datetime(df["p_date"])
        return df
