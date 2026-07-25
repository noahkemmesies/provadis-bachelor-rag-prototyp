"""LLM Integration with Ollama"""

from typing import Optional

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_RETRIES, LLM_MODEL, LLM_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.utils.logger import logger


class OllamaLLM:
    """Integration with Ollama LLM service"""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_TIMEOUT,
    ):
        """
        Initialize Ollama LLM

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name (default: mistral:7b)
            temperature: Temperature for generation (0-2, default: 0.0)
            max_tokens: Max tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_endpoint = f"{self.base_url}/api/generate"

        logger.info(f"Initialized OllamaLLM with model: {model}")

    def check_connection(self) -> bool:
        """
        Check if Ollama server is running

        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            is_connected = response.status_code == 200
            if is_connected:
                logger.info(f"✓ Connected to Ollama at {self.base_url}")
            else:
                logger.warning(f"✗ Ollama returned status {response.status_code}")
            return is_connected
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to Ollama at {self.base_url}")
            logger.error("   Make sure Ollama is running: ollama serve")
            return False
        except Exception as e:
            logger.error(f"✗ Error checking connection: {e}")
            return False

    def list_models(self) -> list:
        """
        List available models in Ollama

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", [])]
                logger.info(f"Available models: {models}")
                return models
            else:
                logger.warning(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response from LLM

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt for context

        Returns:
            Generated text
        """
        # Build full prompt with system context
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.error(f"LLM Error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return f"Error: LLM returned status {response.status_code}"

        except requests.exceptions.Timeout:
            logger.error(f"LLM timeout (>{self.timeout}s)")
            return "Error: LLM request timeout"
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            return "Error: Cannot connect to Ollama"
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error: {str(e)}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Generate response from LLM (streaming)

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt

        Yields:
            Generated text chunks
        """
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
            else:
                logger.error(f"LLM Error: {response.status_code}")
                yield f"Error: {response.status_code}"

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"Error: {str(e)}"
