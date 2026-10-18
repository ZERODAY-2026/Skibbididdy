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
            "You are a helpful data assistant powered by Google Gemma. "
            "Your task is to write a valid, read-only Snowflake SQL query to answer the user's question.\n\n"
            "Rules:\n"
            "1. Output only the SQL query inside a ```sql ... ``` code block.\n"
            "2. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or administrative queries.\n"
            "3. Use standard Snowflake SQL syntax.\n"
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
        """Extract clean SQL string from model output."""
        match = re.search(r"```sql\s*(SELECT[\s\S]+?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match_general = re.search(r"(SELECT\s+[\s\S]+?;?)", text, re.IGNORECASE)
        if match_general:
            return match_general.group(1).strip().rstrip(";")
        return text.strip()
