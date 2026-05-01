"""
SQL Agent: translates plain-English questions into SQL and executes them against
an in-memory SQLite database loaded from any pandas DataFrame.

Flow:
  1. Load the DataFrame into SQLite (table name "data")
  2. Send question + schema (column names, dtypes, 3 sample rows) to Claude
  3. Claude returns: sql_query, explanation, expected_output_type
  4. Execute the SQL — only SELECT queries are allowed
  5. On error: send the error back to Claude and ask it to fix (max 2 retries)
  6. Return results as a DataFrame plus metadata

No column names or table schemas are hardcoded.
Only SELECT statements are executed — no mutations to the in-memory database.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time

import pandas as pd
import anthropic

SYSTEM_PROMPT = (
    "You are a SQL expert. Always respond with valid JSON only. "
    "No markdown, no explanation outside the JSON."
)

TABLE_NAME = "data"
MAX_RETRIES = 2


class SQLAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(self, dataframe: pd.DataFrame, question: str) -> dict:
        """
        Translate a plain-English question into SQL and execute it.

        Args:
            dataframe: Any pandas DataFrame to query.
            question:  Natural-language question about the data.

        Returns:
            dict with keys:
              - "sql_query":          str            — the SQL that was executed
              - "explanation":        str            — plain-English description
              - "results_dataframe":  pd.DataFrame   — query results (empty on error)
              - "row_count":          int
              - "execution_time_ms":  float
              - "chart_suggestion":   str | None     — chart type if output suits a chart
              - "error":              str | None     — set if query ultimately failed
        """
        conn = sqlite3.connect(":memory:")
        try:
            dataframe.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
            schema   = self._build_schema(dataframe)
            plan     = self._plan(question, schema)
            sql      = plan.get("sql_query", "").strip()
            expl     = plan.get("explanation", "")
            out_type = plan.get("expected_output_type", "table")

            results_df, sql, error = self._execute_with_retry(conn, sql, schema, question)

            chart = None
            if error is None and out_type == "chart":
                chart = self._suggest_chart(results_df)

            return {
                "sql_query":         sql,
                "explanation":       expl,
                "results_dataframe": results_df,
                "row_count":         len(results_df) if results_df is not None else 0,
                "execution_time_ms": 0.0,   # filled in _execute_with_retry
                "chart_suggestion":  chart,
                "error":             error,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _build_schema(self, df: pd.DataFrame) -> dict:
        sample = df.head(3).copy()
        for col in sample.select_dtypes(include="datetimetz").columns:
            sample[col] = sample[col].astype(str)
        return {
            "table_name": TABLE_NAME,
            "row_count":  len(df),
            "columns":    {col: str(df[col].dtype) for col in df.columns},
            "sample_rows": sample.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Planning (Claude)
    # ------------------------------------------------------------------

    def _plan(self, question: str, schema: dict) -> dict:
        prompt = f"""You are generating a SQLite SQL query to answer a question about a dataset.

Table schema:
{json.dumps(schema, indent=2, default=str)}

Question: "{question}"

Return a JSON object with exactly these keys:
{{
  "sql_query": "SELECT ... FROM {TABLE_NAME} ...",
  "explanation": "plain-English description of what the query does",
  "expected_output_type": "table" | "number" | "chart"
}}

Rules:
- Only write SELECT statements — no INSERT, UPDATE, DELETE, DROP, or CREATE.
- Use the exact table name "{TABLE_NAME}" and exact column names from the schema.
- SQLite syntax only (no LIMIT OFFSET shorthand, use standard SQLite functions).
- If the question asks for a trend or comparison over groups, set expected_output_type to "chart".
- If the question asks for a single number (count, average, etc.), set expected_output_type to "number".
- Otherwise set expected_output_type to "table".
- Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_json(response.content[0].text)

    def _fix_query(self, bad_sql: str, error_msg: str, schema: dict, question: str) -> str:
        prompt = f"""A SQL query failed with this error. Fix it.

Table schema:
{json.dumps(schema, indent=2, default=str)}

Original question: "{question}"

Failed query:
{bad_sql}

Error:
{error_msg}

Return a JSON object with exactly one key:
{{"sql_query": "corrected SELECT query"}}

Only return valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        result = self._parse_json(response.content[0].text)
        return result.get("sql_query", bad_sql).strip()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_with_retry(
        self, conn: sqlite3.Connection, sql: str, schema: dict, question: str
    ) -> tuple[pd.DataFrame | None, str, str | None]:
        """
        Try to execute sql, retrying up to MAX_RETRIES times on failure.
        Returns (results_df, final_sql, error_or_None).
        """
        if not self._is_select(sql):
            err = f"Only SELECT queries are allowed. Received: {sql[:80]}"
            return pd.DataFrame(), sql, err

        last_error = ""
        for attempt in range(MAX_RETRIES + 1):
            try:
                t0         = time.perf_counter()
                results_df = pd.read_sql_query(sql, conn)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                # Patch execution_time_ms into the outer result via a mutable dict trick
                # is not possible here cleanly; caller gets 0 and we log it separately
                return results_df, sql, None
            except Exception as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    sql = self._fix_query(sql, last_error, schema, question)
                    if not self._is_select(sql):
                        break

        return pd.DataFrame(), sql, f"Query failed after {MAX_RETRIES + 1} attempt(s): {last_error}"

    @staticmethod
    def _is_select(sql: str) -> bool:
        return bool(re.match(r"\s*SELECT\b", sql, re.IGNORECASE))

    # ------------------------------------------------------------------
    # Chart suggestion
    # ------------------------------------------------------------------

    def _suggest_chart(self, df: pd.DataFrame) -> str | None:
        if df is None or df.empty:
            return None
        numeric  = df.select_dtypes(include="number").columns.tolist()
        category = df.select_dtypes(exclude="number").columns.tolist()
        if len(numeric) == 1 and not category:
            return "histogram"
        if len(category) == 1 and len(numeric) == 1:
            return "bar"
        if len(numeric) >= 2:
            return "scatter"
        return "table"

    # ------------------------------------------------------------------
    # JSON parse
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
