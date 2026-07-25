"""Text preprocessing utilities"""

import re
import string
from typing import List

from src.config import (
    LOWERCASE,
    MIN_TOKEN_LENGTH,
    REMOVE_EXTRA_WHITESPACE,
    REMOVE_PUNCTUATION,
)
from src.utils.logger import logger


class TextPreprocessor:
    """Preprocess text for retrieval tasks"""

    def __init__(self, lowercase: bool = LOWERCASE,
                 remove_punctuation: bool = REMOVE_PUNCTUATION,
                 remove_extra_whitespace: bool = REMOVE_EXTRA_WHITESPACE,
                 min_token_length: int = MIN_TOKEN_LENGTH):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_token_length = min_token_length

    def preprocess(self, text: str) -> str:
        """
        Preprocess text with configured options

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = " ".join(text.split())

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        preprocessed = self.preprocess(text)
        tokens = preprocessed.split()

        # Filter by min length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def preprocess_documents(self, documents: List[dict]) -> List[dict]:
        """
        Preprocess a list of documents

        Args:
            documents: List of document dicts with 'content' key

        Returns:
            List of documents with preprocessed 'content' key
        """
        preprocessed_docs = []

        for doc in documents:
            doc_copy = doc.copy()
            doc_copy["content"] = self.preprocess(doc["content"])
            preprocessed_docs.append(doc_copy)

        logger.info(f"Preprocessed {len(preprocessed_docs)} documents")
        return preprocessed_docs


def remove_yaml_comments(text: str) -> str:
    """Remove YAML comments from text"""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Remove inline comments
        if '#' in line:
            line = line[:line.index('#')]
        cleaned_lines.append(line.rstrip())

    return '\n'.join(cleaned_lines)


def extract_yaml_keys(text: str) -> List[str]:
    """Extract YAML keys/fields from text"""
    keys = []
    for line in text.split('\n'):
        # Match lines like "key:" or "key: value"
        match = re.match(r'^[\s]*([a-zA-Z_][a-zA-Z0-9_\-]*)\s*:', line)
        if match:
            keys.append(match.group(1))
    return keys
