"""Google Gemma LLM client for generating Snowflake SQL queries."""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class GemmaClient:
    """Interfaces with Google Gemma via OpenAI-compatible API endpoint."""

    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/openai/v1")
        self.api_key = os.getenv("LLM_API_KEY", "dummy-key-for-local")
        self.model = os.getenv("LLM_MODEL", "google.gemma-4-31b")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate_sql(self, question: str, schema_context: str = "") -> str:
        """Translate a natural language question into a Snowflake SQL query."""
        system_prompt = (
            "You are an expert SQL engineer and data assistant powered by Google Gemma. "
            "Your task is to write a valid, performant, read-only Snowflake SQL query to answer the user's question.\n\n"
            "Guidelines:\n"
            "1. Output the query inside a ```sql ... ``` code block.\n"
            "2. Read-only strictly: never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE statements.\n"
            "3. Snowflake Dialect rules:\n"
            "   - Use DATE_TRUNC('month', col) or EXTRACT(year FROM col) for dates.\n"
            "   - Use DATEDIFF('day', start_date, end_date) for date arithmetic.\n"
            "   - Use ILIKE for case-insensitive string matching.\n"
            "   - Use ROUND(val, 2) for numeric aggregates.\n"
            "   - Explicitly alias computed columns.\n"
            "4. Only reference tables and columns provided in the Schema Context.\n"
        )
        if schema_context:
            system_prompt += f"\nDatabase Schema Context:\n{schema_context}\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content or ""
        return self._extract_sql(content)

    def _extract_sql(self, text: str) -> str:
        """Extract clean SQL string from model output and strip trailing artifacts."""
        match = re.search(r"```sql\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
        else:
            match_general = re.search(r"((?:WITH|SELECT)\s+[\s\S]+?;?)", text, re.IGNORECASE)
            candidate = match_general.group(1).strip() if match_general else text.strip()

        # Remove trailing markdown or semicolons for uniform execution
        clean_lines = []
        for line in candidate.splitlines():
            line_str = line.strip()
            if line_str.startswith("```"):
                continue
            clean_lines.append(line)

        return "\n".join(clean_lines).rstrip(" ;")
