from typing import Callable, Literal, ParamSpec, TypeVar
from pydantic import BaseModel
import os

class OllamaClientOptions(BaseModel):
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama.webtw.xyz:11434")
    model: str = os.getenv("LLM_MODEL", "llama3.3")

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
        # 將回應解析為指定的回應模型
        return response_model.parse_raw(response)
    
    return wrapper
