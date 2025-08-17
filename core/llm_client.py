import asyncio
import httpx
import json
from typing import Optional, List, Dict, Any

from astrbot.api import logger
from astrbot.core.message.context import Context

class LLMResponse:
    """
    模拟 AstrBot 内部 LLMResponse 的简化类。
    """
    def __init__(self, text: str, raw_response: Dict[str, Any]):
        self._text = text
        self._raw_response = raw_response

    def text(self) -> str:
        return self._text

    def raw(self) -> Dict[str, Any]:
        return self._raw_response

class LLMClient:
    """
    封装自定义 LLM API 调用的客户端。
    用于根据配置的 API URL 和 API Key 调用不同的 LLM。
    """

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.client = httpx.AsyncClient(timeout=60.0) # 设置超时时间

    async def chat_completion(
        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[LLMResponse]:
        """
        执行 LLM 对话补全。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if contexts:
            messages.extend(contexts)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }

        try:
            logger.debug(f"Calling LLM API: {self.api_url} with model {self.model_name}")
            response = await self.client.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status() # 检查HTTP错误

            response_data = response.json()
            
            # 假设LLM响应格式与OpenAI兼容
            if "choices" in response_data and len(response_data["choices"]) > 0:
                text_content = response_data["choices"][0]["message"]["content"]
                return LLMResponse(text=text_content, raw_response=response_data)
            else:
                logger.error(f"LLM API 响应格式不正确或无内容: {response_data}")
                return None
        except httpx.RequestError as e:
            logger.error(f"调用 LLM API ({self.api_url}) 请求失败: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"调用 LLM API ({self.api_url}) HTTP 错误: {e.response.status_code} - {e.response.text}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"LLM API 响应解析失败: {e} - 响应内容: {response.text}")
            return None
        except Exception as e:
            logger.error(f"调用 LLM API ({self.api_url}) 发生未知错误: {e}")
            return None
