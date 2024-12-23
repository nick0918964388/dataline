from typing import Optional, Type
from mirascope import tags
from mirascope.openai import OpenAICallParams, OpenAIExtractor
from pydantic import BaseModel
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging


class SQLCorrectionDetails(BaseModel):
    needs_correction: bool
    query: Optional[str] = None


class OllamaExtractor:
    def __init__(self, model_name="llama2-vision", base_url="http://ollama.webtw.xyz:11434"):
        self.model_name = model_name
        self.base_url = base_url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")
            return response.json()["response"]
        except Exception as e:
            logging.error(f"Error calling Ollama API: {str(e)}")
            raise


@tags(["version:0001"])
class QuerySQLCorrectorCall:
    def __init__(self):
        self.ollama_client = OllamaExtractor()

    async def execute(self, query: str, dialect: str) -> SQLCorrectionDetails:
        prompt = f"""
        {query}

        請使用繁體中文回應。
        我應該要確保這個查詢是正確的以獲得獎勵！
        請檢查上面的 {dialect} 查詢是否有常見錯誤，包括：
    - 在 NOT IN 中使用 NULL 值
    - 應該使用 UNION ALL 時卻使用了 UNION
    - 使用 BETWEEN 進行排他範圍查詢
    - 謂詞中的資料類型不匹配
    - 正確引用標識符
    - 函數使用的參數數量是否正確
    - 轉換為正確的資料類型
    - 使用正確的欄位進行連接
    - 使用模糊的欄位名稱（這是你經常犯的錯誤！）

    如果查詢正確，請將 needs_correction 設為 False，並且不要重複填寫查詢。
    如果查詢需要修正，請將 needs_correction 設為 True，並填入修正後的查詢。
        """
        
        response = await self.ollama_client.generate(prompt)
        # 解析回應並轉換成 SQLCorrectionDetails 格式
        # ... 實作解析邏輯
