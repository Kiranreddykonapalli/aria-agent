"""
Data Prep Agent: applies user-specified cleaning operations to any DataFrame.

The user describes what they want in plain English. Claude translates the
instruction into a structured JSON plan of pandas operations. Each operation
is then executed safely — no eval(), no exec(), only direct pandas API calls.

Supported operations:
  drop_columns       — remove one or more columns
  drop_rows_where    — remove rows matching a condition
  fill_nulls         — fill missing values in a column
  rename_column      — rename a column
  filter_rows        — keep only rows matching a condition
  remove_duplicates  — drop exact duplicate rows
  convert_dtype      — cast a column to int / float / str / datetime
  normalize_column   — min-max scale a numeric column to [0, 1]

Returns the cleaned DataFrame, a before/after summary, and the saved CSV path.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone

import pandas as pd
import anthropic

OUTPUT_DIR = "output/data_prep"

SYSTEM_PROMPT = (
    "You are a data engineering assistant. Always respond with valid JSON only. "
    "No markdown, no explanation outside the JSON."
)

VALID_OPS = {
    "drop_columns", "drop_rows_where", "fill_nulls", "rename_column",
    "filter_rows", "remove_duplicates", "convert_dtype", "normalize_column",
}
VALID_OPERATORS  = {"==", "!=", ">", "<", ">=", "<=", "isnull", "notnull"}
VALID_DTYPES     = {"int", "float", "str", "datetime"}


class DataPrepAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run(self, dataframe: pd.DataFrame, user_instruction: str) -> dict:
        """
        Apply user-specified cleaning operations to the DataFrame.

        Args:
            dataframe:        Input DataFrame (not modified in-place).
            user_instruction: Plain-English description of what to clean.

        Returns:
            dict with keys:
              - "dataframe":         pd.DataFrame — cleaned result
              - "operations_applied": list[dict]  — ops that ran successfully
              - "operations_skipped": list[dict]  — ops skipped due to errors
              - "before_shape":      tuple(int, int)
              - "after_shape":       tuple(int, int)
              - "output_path":       str — saved CSV path
              - "explanation":       str — Claude's plain-English plan description
              - "warnings":          list[str] — risks Claude flagged
        """
        before_shape = dataframe.shape
        plan = self._plan(dataframe, user_instruction)

        df = dataframe.copy()
        ops_applied: list[dict] = []
        ops_skipped: list[dict] = []

        for op in plan.get("operations", []):
            try:
                df = self._execute(df, op)
                ops_applied.append(op)
            except Exception as exc:
                op_copy = dict(op)
                op_copy["error"] = str(exc)
                ops_skipped.append(op_copy)

        output_path = self._save(df)

        return {
            "dataframe":          df,
            "operations_applied": ops_applied,
            "operations_skipped": ops_skipped,
            "before_shape":       before_shape,
            "after_shape":        df.shape,
            "output_path":        output_path,
            "explanation":        plan.get("explanation", ""),
            "warnings":           plan.get("warnings", []),
        }

    # ------------------------------------------------------------------
    # Planning (Claude)
    # ------------------------------------------------------------------

    def _plan(self, df: pd.DataFrame, instruction: str) -> dict:
        col_info = {
            col: {
                "dtype":      str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "sample":     df[col].dropna().head(3).tolist(),
            }
            for col in df.columns
        }

        prompt = f"""The user wants to clean a dataset with {df.shape[0]} rows and {df.shape[1]} columns.

User instruction: "{instruction}"

Column information:
{json.dumps(col_info, indent=2, default=str)}

Return a JSON object with exactly these keys:

{{
  "operations": [
    // list of operations to perform, in order
  ],
  "explanation": "plain-English description of what will happen",
  "warnings": ["any risks or caveats the user should know"]
}}

Each operation must be one of these formats:

{{"type": "drop_columns",    "columns": ["col1", "col2"]}}
{{"type": "drop_rows_where", "column": "col", "operator": "=="|"!="|">"|"<"|">="|"<="|"isnull"|"notnull", "value": <val_or_null>}}
{{"type": "fill_nulls",      "column": "col", "value": <scalar>}}
{{"type": "rename_column",   "old_name": "col", "new_name": "new_col"}}
{{"type": "filter_rows",     "column": "col", "operator": "=="|"!="|">"|"<"|">="|"<="|"isnull"|"notnull", "value": <val_or_null>}}
{{"type": "remove_duplicates"}}
{{"type": "convert_dtype",   "column": "col", "to": "int"|"float"|"str"|"datetime"}}
{{"type": "normalize_column","column": "col"}}

Rules:
- Only reference columns that exist in the dataset above.
- Only use operators from the allowed list.
- Keep the operations list minimal — only what is needed for the user's instruction.
- Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        raw     = response.content[0].text
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DataPrepAgent: failed to parse Claude plan as JSON.\n"
                f"Error: {exc}\nRaw: {raw[:400]}"
            ) from exc

    # ------------------------------------------------------------------
    # Execution (pandas only — no eval/exec)
    # ------------------------------------------------------------------

    def _execute(self, df: pd.DataFrame, op: dict) -> pd.DataFrame:
        op_type = op.get("type", "")

        if op_type not in VALID_OPS:
            raise ValueError(f"Unknown operation type: '{op_type}'")

        if op_type == "drop_columns":
            cols = op["columns"]
            self._assert_columns_exist(df, cols)
            return df.drop(columns=cols)

        if op_type == "remove_duplicates":
            return df.drop_duplicates()

        if op_type == "fill_nulls":
            col = op["column"]
            self._assert_columns_exist(df, [col])
            return df.assign(**{col: df[col].fillna(op["value"])})

        if op_type == "rename_column":
            old, new = op["old_name"], op["new_name"]
            self._assert_columns_exist(df, [old])
            return df.rename(columns={old: new})

        if op_type == "normalize_column":
            col = op["column"]
            self._assert_columns_exist(df, [col])
            s = pd.to_numeric(df[col], errors="coerce")
            lo, hi = s.min(), s.max()
            if hi == lo:
                raise ValueError(f"normalize_column: '{col}' has zero range — cannot normalise.")
            return df.assign(**{col: (s - lo) / (hi - lo)})

        if op_type == "convert_dtype":
            col  = op["column"]
            to   = op["to"]
            if to not in VALID_DTYPES:
                raise ValueError(f"convert_dtype: unsupported dtype '{to}'.")
            self._assert_columns_exist(df, [col])
            if to == "int":
                return df.assign(**{col: pd.to_numeric(df[col], errors="coerce").astype("Int64")})
            if to == "float":
                return df.assign(**{col: pd.to_numeric(df[col], errors="coerce")})
            if to == "str":
                return df.assign(**{col: df[col].astype(str)})
            if to == "datetime":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return df.assign(**{col: pd.to_datetime(df[col], errors="coerce")})

        # drop_rows_where and filter_rows share mask logic
        if op_type in ("drop_rows_where", "filter_rows"):
            col      = op["column"]
            operator = op["operator"]
            value    = op.get("value")
            self._assert_columns_exist(df, [col])
            if operator not in VALID_OPERATORS:
                raise ValueError(f"Invalid operator '{operator}'.")
            mask = self._build_mask(df[col], operator, value)
            if op_type == "drop_rows_where":
                return df[~mask]
            else:  # filter_rows — keep matching
                return df[mask]

        raise ValueError(f"Unhandled operation: {op_type}")

    def _build_mask(self, series: pd.Series, operator: str, value) -> pd.Series:
        if operator == "isnull":
            return series.isnull()
        if operator == "notnull":
            return series.notnull()
        numeric_series = pd.to_numeric(series, errors="ignore")
        try:
            value = type(numeric_series.dropna().iloc[0])(value)
        except Exception:
            pass
        ops = {
            "==": numeric_series == value,
            "!=": numeric_series != value,
            ">":  numeric_series > value,
            "<":  numeric_series < value,
            ">=": numeric_series >= value,
            "<=": numeric_series <= value,
        }
        return ops[operator]

    @staticmethod
    def _assert_columns_exist(df: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in dataframe: {missing}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path      = os.path.join(OUTPUT_DIR, f"{timestamp}_cleaned.csv")
        df.to_csv(path, index=False)
        return path
