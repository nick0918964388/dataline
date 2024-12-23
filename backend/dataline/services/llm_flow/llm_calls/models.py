from pydantic import BaseModel
from langchain_core.messages import ToolCall


class OllamaResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = []

    @classmethod
    def parse_tool_calls(cls, raw_tool_calls: list[dict]) -> list[ToolCall]:
        return [
            ToolCall(
                id=f"call_{i}",
                type="function",
                function=dict(
                    name=call.get("function", {}).get("name", ""),
                    arguments=call.get("function", {}).get("arguments", {})
                )
            )
            for i, call in enumerate(raw_tool_calls)
        ] 