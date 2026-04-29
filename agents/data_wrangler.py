"""
Data Wrangler: loads raw healthcare data and prepares it for analysis.

Responsibilities:
  - Load CSV from data/raw/
  - Validate schema (required columns present)
  - Enforce correct dtypes (year/population as int, rates as float)
  - Check for nulls and duplicate rows
  - Flag suspicious values (rates outside 0-1, negative populations)
  - Emit a data_quality_report dict summarising all findings
  - Save cleaned data to data/processed/florida_health_clean.csv
  - Return dict with keys: dataframe, data_quality_report, file_path

No Claude API calls — pure pandas.
Raw source files are never modified.
"""

import os

import pandas as pd

EXPECTED_COLUMNS: list[str] = [
    "county",
    "year",
    "population",
    "uninsured_rate",
    "obesity_rate",
    "diabetes_rate",
    "mental_health_days",
    "physical_health_days",
    "primary_care_physicians_rate",
    "median_household_income",
    "high_school_graduation_rate",
    "unemployment_rate",
    "health_outcome_rank",
    "health_factor_rank",
]

# Columns that must sit in [0, 1] (they are proportions/rates).
RATE_COLUMNS: list[str] = [
    "uninsured_rate",
    "obesity_rate",
    "diabetes_rate",
    "high_school_graduation_rate",
    "unemployment_rate",
]

INT_COLUMNS: list[str] = ["year", "population", "health_outcome_rank", "health_factor_rank"]
FLOAT_COLUMNS: list[str] = [
    "uninsured_rate",
    "obesity_rate",
    "diabetes_rate",
    "mental_health_days",
    "physical_health_days",
    "primary_care_physicians_rate",
    "median_household_income",
    "high_school_graduation_rate",
    "unemployment_rate",
]

OUTPUT_FILENAME = "florida_health_clean.csv"


class DataWrangler:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

    def run(self, raw_path: str) -> dict:
        """
        Load, validate, and clean the raw healthcare CSV.

        Args:
            raw_path: Path to the raw CSV file.

        Returns:
            dict with keys:
              - "dataframe":          pd.DataFrame — cleaned dataset
              - "data_quality_report": dict — full quality summary
              - "file_path":          str — path to saved cleaned CSV
        """
        df = self._load(raw_path)
        self._validate_schema(df)
        df, report = self._clean(df)
        file_path = self._save(df)
        report["output_file"] = file_path
        return {"dataframe": df, "data_quality_report": report, "file_path": file_path}

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _load(self, path: str) -> pd.DataFrame:
        """Load CSV into a DataFrame."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")
        if not path.endswith(".csv"):
            raise ValueError(f"Expected a CSV file, got: {path}")
        df = pd.read_csv(path)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Raise ValueError listing every missing required column."""
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Run all cleaning steps in sequence and build the quality report.
        Steps are applied in order; each step records its findings.
        """
        report: dict = {
            "raw_row_count": len(df),
            "raw_column_count": len(df.columns),
            "null_count_by_column": {},
            "nulls_dropped": 0,
            "duplicate_rows_dropped": 0,
            "dtype_coercions": [],
            "suspicious_values": [],
            "final_row_count": 0,
            "final_column_count": 0,
        }

        df = df.copy()
        df = self._check_nulls(df, report)
        df = self._drop_duplicates(df, report)
        df = self._enforce_dtypes(df, report)
        self._flag_suspicious(df, report)

        report["final_row_count"] = len(df)
        report["final_column_count"] = len(df.columns)
        return df, report

    def _check_nulls(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Record null counts per column, then drop any rows that have nulls."""
        null_counts = df.isnull().sum()
        report["null_count_by_column"] = null_counts[null_counts > 0].to_dict()
        total_nulls = int(null_counts.sum())

        if total_nulls > 0:
            before = len(df)
            df = df.dropna()
            report["nulls_dropped"] = before - len(df)
        return df

    def _drop_duplicates(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Drop exact duplicate rows; a (county, year) pair must be unique."""
        before = len(df)
        df = df.drop_duplicates()

        # Also enforce (county, year) uniqueness — keep first occurrence.
        df = df.drop_duplicates(subset=["county", "year"], keep="first")
        report["duplicate_rows_dropped"] = before - len(df)
        return df

    def _enforce_dtypes(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Coerce columns to their expected dtypes, recording any that needed fixing."""
        for col in INT_COLUMNS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                report["dtype_coercions"].append(f"{col} -> Int64")

        for col in FLOAT_COLUMNS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                report["dtype_coercions"].append(f"{col} -> float")

        if "county" in df.columns:
            df["county"] = df["county"].astype(str).str.strip()

        return df

    def _flag_suspicious(self, df: pd.DataFrame, report: dict) -> None:
        """
        Identify values that are technically present but statistically implausible.
        Findings are appended to report["suspicious_values"] as description strings.
        Does not drop rows — flags only.
        """
        issues: list[str] = []

        # Rates must be strictly between 0 and 1.
        for col in RATE_COLUMNS:
            if col not in df.columns:
                continue
            bad = df[(df[col] < 0) | (df[col] > 1)]
            if not bad.empty:
                issues.append(
                    f"{col}: {len(bad)} value(s) outside [0, 1] "
                    f"(min={df[col].min():.4f}, max={df[col].max():.4f})"
                )

        # Population must be positive.
        if "population" in df.columns:
            neg_pop = df[df["population"] <= 0]
            if not neg_pop.empty:
                issues.append(
                    f"population: {len(neg_pop)} non-positive value(s) — "
                    f"counties: {neg_pop['county'].tolist()}"
                )

        # mental_health_days and physical_health_days should be in [0, 30].
        for col in ("mental_health_days", "physical_health_days"):
            if col not in df.columns:
                continue
            bad = df[(df[col] < 0) | (df[col] > 30)]
            if not bad.empty:
                issues.append(f"{col}: {len(bad)} value(s) outside [0, 30]")

        # primary_care_physicians_rate (per 100k) should be > 0.
        if "primary_care_physicians_rate" in df.columns:
            bad = df[df["primary_care_physicians_rate"] < 0]
            if not bad.empty:
                issues.append(
                    f"primary_care_physicians_rate: {len(bad)} negative value(s)"
                )

        # Health ranks must be between 1 and 67.
        for col in ("health_outcome_rank", "health_factor_rank"):
            if col not in df.columns:
                continue
            bad = df[(df[col] < 1) | (df[col] > 67)]
            if not bad.empty:
                issues.append(f"{col}: {len(bad)} value(s) outside [1, 67]")

        # Year must be in expected range.
        if "year" in df.columns:
            bad = df[(df["year"] < 2000) | (df["year"] > 2100)]
            if not bad.empty:
                issues.append(f"year: {len(bad)} value(s) outside plausible range")

        report["suspicious_values"] = issues if issues else ["none detected"]

    def _save(self, df: pd.DataFrame) -> str:
        """Write cleaned DataFrame to data/processed/. Returns the saved path."""
        os.makedirs(self.processed_dir, exist_ok=True)
        path = os.path.join(self.processed_dir, OUTPUT_FILENAME)
        df.to_csv(path, index=False)
        return path
