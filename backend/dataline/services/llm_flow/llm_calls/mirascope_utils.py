from typing import Callable, Literal, ParamSpec, TypeVar
from pydantic import BaseModel
import os
import re
from dataline.errors import UserFacingError
import logging
import json
from .models import OllamaResponse

_T = TypeVar("_T", bound=BaseModel)
P = ParamSpec("P")

logger = logging.getLogger(__name__)

class OllamaClientOptions(BaseModel):
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama.webtw.xyz:11434")
    model: str = os.getenv("LLM_MODEL", "llama3.3-extra")

def call(
    model: str,
    response_model: type[_T],
    prompt_fn: Callable[P, str],
    client_options: OllamaClientOptions,
) -> Callable[P, _T]:
    from .query_sql_corrector import OllamaExtractor
    
    ollama_client = OllamaExtractor(
        model_name=client_options.model,
        base_url=client_options.base_url
    )
    
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> _T:
        prompt = prompt_fn(*args, **kwargs)
        response = await ollama_client.generate(prompt)
        try:
            # 使用正則表達式找出 JSON 部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    parsed["tool_calls"] = OllamaResponse.parse_tool_calls(parsed["tool_calls"])
                return response_model.model_validate(parsed)
            else:
                # 如果找不到 JSON，嘗試將純文本回包裝成所需的格式
                if hasattr(response_model, "title"):
                    return response_model.model_validate({"title": response.strip()})
                else:
                    return response_model.model_validate({
                        "content": response,
                        "tool_calls": []
                    })
        except Exception as e:
            logger.error(f"Error parsing Ollama response: {str(e)}")
            raise UserFacingError(f"無法解析 AI 回應：{str(e)}")
    
    return wrapper
