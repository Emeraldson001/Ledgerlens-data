"""Built-in importers, registered (and conformance-checked) on import."""

import pandas as pd

from ingestion import importer_registry


class HistoricalFileImporter:
    """Loads historical trades from a CSV/Parquet file (pre-parse scanned)."""

    name = "historical_file"
    capabilities = {"load_dataframe"}

    def __init__(self, path: str | None = None):
        self.path = path

    def load_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        if self.path is None:
            return pd.DataFrame()
        from ingestion.historical_loader import load_trades_file

        df = load_trades_file(self.path)
        return df.head(limit) if limit else df


def register_builtin_importers() -> None:
    importer_registry.register(HistoricalFileImporter())


register_builtin_importers()
