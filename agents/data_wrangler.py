"""
Data Wrangler: loads, validates, and cleans any tabular CSV dataset.

Responsibilities:
  - Load any CSV file
  - Validate minimum shape (>= 2 columns, >= 10 rows)
  - Auto-detect and coerce column types (numeric, datetime, categorical)
  - Drop null rows and exact duplicate rows
  - Flag data quality issues (high-null columns, constant columns,
    negative values, suspicious proportions)
  - Emit a data_quality_report dict summarising all findings
  - Save cleaned data to data/processed/<input_stem>_clean.csv
  - Return dict with keys: dataframe, data_quality_report, file_path

No Claude API calls — pure pandas.
Raw source files are never modified.
Works with any tabular dataset — no column names are hardcoded.
"""

import os
import warnings
from pathlib import Path

import pandas as pd

MIN_COLUMNS = 2
MIN_ROWS    = 10

# Columns where >N% of values are null are flagged in the report.
HIGH_NULL_THRESHOLD = 0.20

# Minimum fraction of values that must parse successfully to accept
# a type coercion (datetime or numeric).
COERCE_ACCEPT_RATE = 0.80


class DataWrangler:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

    def run(self, raw_path: str) -> dict:
        """
        Load, validate, and clean a CSV file.

        Args:
            raw_path: Path to any CSV file.

        Returns:
            dict with keys:
              - "dataframe":           pd.DataFrame — cleaned dataset
              - "data_quality_report": dict         — full quality summary
              - "file_path":           str          — path to saved cleaned CSV
        """
        df = self._load(raw_path)
        self._validate_shape(df)
        df, report = self._clean(df)
        file_path = self._save(df, raw_path)
        report["output_file"] = file_path
        return {"dataframe": df, "data_quality_report": report, "file_path": file_path}

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _load(self, path: str) -> pd.DataFrame:
        """Load a CSV into a DataFrame."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        if not path.lower().endswith(".csv"):
            raise ValueError(f"Expected a .csv file, got: {path}")
        return pd.read_csv(path)

    def _validate_shape(self, df: pd.DataFrame) -> None:
        """Reject files that are too small to analyse meaningfully."""
        if df.shape[1] < MIN_COLUMNS:
            raise ValueError(
                f"Dataset has {df.shape[1]} column(s); "
                f"at least {MIN_COLUMNS} required."
            )
        if df.shape[0] < MIN_ROWS:
            raise ValueError(
                f"Dataset has {df.shape[0]} row(s); "
                f"at least {MIN_ROWS} required."
            )

    def _clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Run all cleaning steps and build the quality report."""
        report: dict = {
            "raw_row_count":         int(len(df)),
            "raw_column_count":      int(len(df.columns)),
            "null_count_by_column":  {},
            "high_null_columns":     {},
            "nulls_dropped":         0,
            "duplicate_rows_dropped": 0,
            "dtype_coercions":       [],
            "column_types_detected": {},
            "suspicious_values":     [],
            "final_row_count":       0,
            "final_column_count":    0,
        }

        df = df.copy()
        df = self._check_nulls(df, report)
        df = self._drop_duplicates(df, report)
        df = self._detect_and_coerce_types(df, report)
        self._flag_suspicious(df, report)

        report["final_row_count"]   = int(len(df))
        report["final_column_count"] = int(len(df.columns))
        return df, report

    def _check_nulls(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """
        Record per-column null counts and flag heavily-null columns,
        then drop all rows containing any null.
        """
        null_pct   = df.isnull().mean()
        null_counts = df.isnull().sum()

        report["null_count_by_column"] = {
            col: int(n) for col, n in null_counts.items() if n > 0
        }
        report["high_null_columns"] = {
            col: f"{pct:.0%}"
            for col, pct in null_pct.items()
            if pct > HIGH_NULL_THRESHOLD
        }

        if int(null_counts.sum()) > 0:
            before = len(df)
            df = df.dropna()
            report["nulls_dropped"] = before - len(df)

        return df

    def _drop_duplicates(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Drop exact duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        report["duplicate_rows_dropped"] = before - len(df)
        return df

    def _detect_and_coerce_types(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """
        Auto-detect the best dtype for every column.

        Priority order for object/string columns:
          1. datetime  — if >= 80 % of values parse with pd.to_datetime
          2. numeric   — if >= 80 % of values parse with pd.to_numeric;
                         downcast to integer when all values are whole numbers
          3. categorical — leave as string, strip whitespace

        Already-numeric columns are kept as-is but downcast to int
        when all values are whole numbers (no fractional part).
        """
        detected: dict[str, str] = {}

        for col in df.columns:
            series = df[col]

            # ── Already datetime ─────────────────────────────────────
            if pd.api.types.is_datetime64_any_dtype(series):
                detected[col] = "datetime"
                continue

            # ── Already numeric ──────────────────────────────────────
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0 and (non_null % 1 == 0).all():
                    df[col] = series.astype("Int64")
                    detected[col] = "integer"
                    report["dtype_coercions"].append(f"{col}: float -> Int64")
                else:
                    detected[col] = "numeric"
                continue

            # ── Object: strip whitespace ─────────────────────────────
            df[col] = series.astype(str).str.strip()
            series  = df[col]

            # ── Try datetime ─────────────────────────────────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_dt = pd.to_datetime(series, errors="coerce")
            dt_success = parsed_dt.notna().mean()
            if dt_success >= COERCE_ACCEPT_RATE:
                df[col]     = parsed_dt
                detected[col] = "datetime"
                report["dtype_coercions"].append(f"{col}: object -> datetime")
                continue

            # ── Try numeric ──────────────────────────────────────────
            parsed_num  = pd.to_numeric(series, errors="coerce")
            num_success = parsed_num.notna().mean()
            if num_success >= COERCE_ACCEPT_RATE:
                non_null = parsed_num.dropna()
                if len(non_null) > 0 and (non_null % 1 == 0).all():
                    df[col]     = parsed_num.astype("Int64")
                    detected[col] = "integer"
                    report["dtype_coercions"].append(f"{col}: object -> Int64")
                else:
                    df[col]     = parsed_num
                    detected[col] = "numeric"
                    report["dtype_coercions"].append(f"{col}: object -> float")
                continue

            # ── Categorical ──────────────────────────────────────────
            detected[col] = "categorical"

        report["column_types_detected"] = detected
        return df

    def _flag_suspicious(self, df: pd.DataFrame, report: dict) -> None:
        """
        Generic suspicious-value checks that work for any dataset.
        Appends human-readable strings to report["suspicious_values"].
        Rows are never dropped — these are flags only.
        """
        issues: list[str] = []

        # Constant columns (one unique value — useless for analysis)
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                issues.append(
                    f"'{col}' has only {df[col].nunique()} unique value(s) — "
                    "may be constant or all-null after cleaning"
                )

        numeric_cols = df.select_dtypes(include="number").columns

        # Columns with any negative values (flag for review — may be legitimate)
        for col in numeric_cols:
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                issues.append(
                    f"'{col}' contains {n_neg} negative value(s) "
                    f"(min = {df[col].min():.4g})"
                )

        # Columns that look like proportions (most values 0-1) but leak outside
        for col in numeric_cols:
            s        = df[col].dropna()
            in_range = ((s >= 0) & (s <= 1)).mean()
            # Smells like a proportion column but isn't fully clean
            if 0.70 < in_range < 1.0:
                n_out = int(((s < 0) | (s > 1)).sum())
                issues.append(
                    f"'{col}' appears to be a proportion "
                    f"but has {n_out} value(s) outside [0, 1]"
                )

        report["suspicious_values"] = issues if issues else ["none detected"]

    def _save(self, df: pd.DataFrame, raw_path: str) -> str:
        """
        Save cleaned DataFrame to data/processed/<input_stem>_clean.csv.
        Output filename is derived from the input filename so any dataset
        gets a distinct clean file.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        stem = Path(raw_path).stem
        path = os.path.join(self.processed_dir, f"{stem}_clean.csv")
        df.to_csv(path, index=False)
        return path
