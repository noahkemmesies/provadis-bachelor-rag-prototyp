"""Helper functions and utilities"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger import logger


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save data to JSON file"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        raise


def save_pickle(data: Any, filepath: Path) -> None:
    """Save data to pickle file"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved pickle to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle to {filepath}: {e}")
        raise


def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {e}")
        raise


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
