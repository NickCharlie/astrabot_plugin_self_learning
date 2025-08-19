import asyncio
import json
from typing import Optional, List, Dict, Any
import aiohttp

from astrbot.api import logger

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

    def __init__(self):
        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60.0)) # 设置超时时间

    async def chat_completion(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[LLMResponse]:
        """
        执行 LLM 对话补全。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if contexts:
            messages.extend(contexts)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            **kwargs
        }

        try:
            logger.debug(f"Calling LLM API: {api_url} with model {model_name}")
            async with self.client.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status() # 检查HTTP错误

                response_data = await response.json()
                
                # 假设LLM响应格式与OpenAI兼容
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    text_content = response_data["choices"][0]["message"]["content"]
                    return LLMResponse(text=text_content, raw_response=response_data)
                else:
                    logger.error(f"LLM API 响应格式不正确或无内容: {response_data}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"调用 LLM API ({api_url}) 请求失败或HTTP错误: {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"LLM API 响应解析失败: {e} - 响应内容: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"调用 LLM API ({api_url}) 发生未知错误: {e}", exc_info=True)
            return None
