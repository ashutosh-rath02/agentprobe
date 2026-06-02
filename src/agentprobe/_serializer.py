from typing import Any, Dict

import openai.types.chat


def _deep_serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_serialize(item) for item in obj]
    return obj


def serialize_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _deep_serialize(v) for k, v in kwargs.items()}


def deserialize_response(data: Dict[str, Any]) -> openai.types.chat.ChatCompletion:
    return openai.types.chat.ChatCompletion.model_validate(data)


def deserialize_chunk(data: Dict[str, Any]) -> openai.types.chat.ChatCompletionChunk:
    return openai.types.chat.ChatCompletionChunk.model_validate(data)
