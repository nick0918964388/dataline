from typing import Optional, Type

from mirascope import tags
from mirascope.openai import OpenAICallParams, OpenAIExtractor
from pydantic import BaseModel


class SQLCorrectionDetails(BaseModel):
    needs_correction: bool
    query: Optional[str] = None


@tags(["version:0001"])
class QuerySQLCorrectorCall(OpenAIExtractor[SQLCorrectionDetails]):
    extract_schema: Type[SQLCorrectionDetails] = SQLCorrectionDetails
    call_params = OpenAICallParams(model="gpt-3.5-turbo")
    api_key: str | None

    prompt_template = """
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

    query: str
    dialect: str
