#!/usr/bin/env python
"""Download required models for embeddings and LLM"""

import subprocess
import sys

from src.utils.logger import logger


def download_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """
    Download Sentence Transformers model

    Args:
        model_name: Model name from Hugging Face

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading Sentence Transformers model: {model_name}...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully downloaded {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return False


def download_ollama_model(model_name: str = "mistral:7b") -> bool:
    """
    Download Ollama model

    Requires Ollama to be installed: https://ollama.ai

    Args:
        model_name: Model name (e.g., "mistral:7b", "llama2:latest")

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading Ollama model: {model_name}...")
        logger.info("Make sure Ollama is installed and the daemon is running.")
        logger.info("Install from: https://ollama.ai")

        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info(f"Successfully downloaded {model_name}")
            return True
        else:
            logger.error(f"Error downloading {model_name}: {result.stderr}")
            return False

    except FileNotFoundError:
        logger.error("Ollama command not found. Please install Ollama from https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading {model_name}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return False


def check_ollama_running() -> bool:
    """
    Check if Ollama daemon is running

    Returns:
        True if running, False otherwise
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def main():
    """Download all required models"""
    logger.info("Starting model download process...")

    # Download Sentence Transformers model
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Downloading Sentence Transformers model")
    logger.info("=" * 60)
    if download_sentence_transformer_model():
        logger.info("✓ Sentence Transformers model downloaded successfully")
    else:
        logger.error("✗ Failed to download Sentence Transformers model")
        sys.exit(1)

    # Check Ollama
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Checking Ollama installation")
    logger.info("=" * 60)

    if not check_ollama_running():
        logger.warning("Ollama daemon is not running!")
        logger.info("Please start Ollama with: ollama serve")
        logger.info("Or install from: https://ollama.ai")
        response = input("Continue without Ollama? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    else:
        logger.info("✓ Ollama daemon is running")

        # Download Ollama model
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Downloading Ollama model")
        logger.info("=" * 60)
        if download_ollama_model():
            logger.info("✓ Ollama model downloaded successfully")
        else:
            logger.error("✗ Failed to download Ollama model")
            logger.info("You can try manually: ollama pull mistral:7b")

    logger.info("\n" + "=" * 60)
    logger.info("Model download process completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
