from pydantic import BaseModel


class ConversationTitleGeneratorResponse(BaseModel):
    title: str


def conversation_title_generator_prompt(user_message: str) -> str:
    return f"""
    請根據以下用戶訊息生成一個簡短的對話標題（不超過 30 個字）：

    {user_message}

    請以 JSON 格式回應，格式如下：
    {{
        "title": "對話標題"
    }}
    """
