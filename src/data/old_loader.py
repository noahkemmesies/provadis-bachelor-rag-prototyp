"""Data loading utilities for YAML documents and queries"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.utils.logger import logger


def load_yaml_documents(data_dir: Path) -> List[Dict[str, str]]:
    """
    Load all YAML documents from a directory

    Args:
        data_dir: Directory containing YAML files

    Returns:
        List of document dictionaries with 'filename' and 'content' keys
    """
    documents = []
    yaml_files = list(data_dir.rglob("*.yaml")) + list(data_dir.rglob("*.yml"))

    if not yaml_files:
        logger.warning(f"No YAML files found in {data_dir}")
        return documents

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)

            # Convert YAML to string representation
            yaml_content = yaml.dump(content, default_flow_style=False)

            documents.append({
                "filename": yaml_file.name,
                "filepath": str(yaml_file),
                "content": yaml_content,
            })
            logger.debug(f"Loaded {yaml_file.name}")

        except Exception as e:
            logger.error(f"Error loading {yaml_file}: {e}")
            continue

    logger.info(f"Loaded {len(documents)} YAML documents from {data_dir}")
    return documents


def load_text_documents(data_dir: Path) -> List[Dict[str, str]]:
    """
    Load all text documents from a directory (txt, md)

    Args:
        data_dir: Directory containing text files

    Returns:
        List of document dictionaries
    """
    documents = []
    text_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))

    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()

            documents.append({
                "filename": text_file.name,
                "filepath": str(text_file),
                "content": content,
            })
            logger.debug(f"Loaded {text_file.name}")

        except Exception as e:
            logger.error(f"Error loading {text_file}: {e}")
            continue

    logger.info(f"Loaded {len(documents)} text documents from {data_dir}")
    return documents


def load_queries(queries_file: Path) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Load queries and ground truth from JSON file

    Expected JSON format:
    {
        "queries": [
            {"id": "q1", "text": "What is a Deployment?"},
            ...
        ],
        "ground_truth": {
            "q1": ["doc1.yaml", "doc2.yaml"],
            ...
        }
    }

    Args:
        queries_file: Path to queries JSON file

    Returns:
        Tuple of (queries list, ground_truth dict)
    """
    if not queries_file.exists():
        logger.warning(f"Queries file not found: {queries_file}")
        return [], {}

    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = data.get("queries", [])
        ground_truth = data.get("ground_truth", {})

        logger.info(f"Loaded {len(queries)} queries with ground truth")
        return queries, ground_truth

    except Exception as e:
        logger.error(f"Error loading queries from {queries_file}: {e}")
        raise


def save_queries(queries: List[Dict], ground_truth: Dict[str, List[str]],
                 output_file: Path) -> None:
    """
    Save queries and ground truth to JSON file

    Args:
        queries: List of query dictionaries
        ground_truth: Mapping of query IDs to relevant document filenames
        output_file: Path to output JSON file
    """
    data = {
        "queries": queries,
        "ground_truth": ground_truth,
    }

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved queries to {output_file}")
    except Exception as e:
        logger.error(f"Error saving queries to {output_file}: {e}")
        raise
