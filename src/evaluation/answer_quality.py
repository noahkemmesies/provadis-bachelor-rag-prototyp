"""Answer quality evaluation for RAG systems"""

from typing import Dict, List

import numpy as np

from src.utils.logger import logger


class AnswerQualityEvaluator:
    """Evaluate quality of generated answers"""

    # Keywords for different answer aspects
    COMPLETENESS_KEYWORDS = {
        "kubernetes": ["pod", "deployment", "service", "container", "yaml"],
        "configuration": ["config", "secret", "volume", "mount", "env"],
        "networking": ["service", "ingress", "network", "dns", "port"],
        "storage": ["volume", "pvc", "storage", "persistent", "mount"],
        "scaling": ["replica", "scale", "autoscal", "hpa", "vpa"],
        "rbac": ["role", "permission", "rbac", "serviceaccount", "cluster"],
    }

    def __init__(self):
        """Initialize Answer Quality Evaluator"""
        logger.info("Initialized AnswerQualityEvaluator")

    def evaluate_answer_length(self, answer: str) -> Dict[str, float]:
        """
        Evaluate answer based on length

        Args:
            answer: Generated answer

        Returns:
            Dict with length metrics
        """
        word_count = len(answer.split())
        char_count = len(answer)

        # Ideal: 50-200 words
        if word_count < 10:
            length_score = 0.2
        elif word_count < 50:
            length_score = 0.7
        elif word_count <= 200:
            length_score = 1.0
        elif word_count <= 400:
            length_score = 0.8
        else:
            length_score = 0.5

        return {
            "word_count": word_count,
            "char_count": char_count,
            "length_score": length_score,
        }

    def evaluate_answer_content(self, answer: str, question: str) -> Dict[str, float]:
        """
        Evaluate answer content quality

        Args:
            answer: Generated answer
            question: Original question

        Returns:
            Dict with content metrics
        """
        answer_lower = answer.lower()
        question_lower = question.lower()

        # Check for question keywords in answer
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())
        keyword_coverage = len(question_words & answer_words) / len(question_words)

        # Check for structure
        has_code = "```" in answer or "`" in answer
        has_lists = "- " in answer or "* " in answer
        has_examples = "example" in answer_lower or "for instance" in answer_lower

        structure_score = (has_code * 0.3 + has_lists * 0.3 + has_examples * 0.4)

        # Check for error indicators
        has_error = any(
            word in answer_lower for word in ["error", "failed", "cannot", "not available"]
        )

        completeness_score = 1.0 if not has_error else 0.5

        return {
            "keyword_coverage": keyword_coverage,
            "has_code": has_code,
            "has_lists": has_lists,
            "has_examples": has_examples,
            "structure_score": structure_score,
            "completeness_score": completeness_score,
        }

    def evaluate_relevance(self, answer: str, retrieved_documents: List[Dict]) -> Dict:
        """
        Evaluate answer relevance to retrieved documents

        Args:
            answer: Generated answer
            retrieved_documents: List of retrieved documents

        Returns:
            Dict with relevance metrics
        """
        answer_lower = answer.lower()

        # Check if answer references retrieved documents
        source_mentions = 0
        for doc in retrieved_documents:
            filename = doc.get("filename", "").lower()
            if filename.replace(".yaml", "").replace("_", " ") in answer_lower:
                source_mentions += 1

        source_citation_score = min(source_mentions / len(retrieved_documents), 1.0) if retrieved_documents else 0.0

        # Check for specific kubernetes terms
        k8s_terms = {
            "pod": 0.2,
            "deployment": 0.2,
            "service": 0.15,
            "statefulset": 0.15,
            "daemonset": 0.1,
            "job": 0.1,
            "ingress": 0.05,
        }

        term_score = 0.0
        for term, weight in k8s_terms.items():
            if term in answer_lower:
                term_score += weight

        term_score = min(term_score, 1.0)

        return {
            "source_mentions": source_mentions,
            "source_citation_score": source_citation_score,
            "k8s_term_coverage": term_score,
        }

    def evaluate_technical_accuracy(self, answer: str) -> Dict[str, float]:
        """
        Evaluate technical accuracy of answer

        Args:
            answer: Generated answer

        Returns:
            Dict with accuracy metrics
        """
        answer_lower = answer.lower()

        # Check for common mistakes
        mistakes = []

        # YAML indentation issues
        if "indent" in answer_lower or "indentation" in answer_lower:
            mistakes.append("mentions indentation")

        # Port issues
        if ("port" in answer_lower) and ("containerport" in answer_lower):
            mistakes.append("mentions containerPort correctly")

        # Resource limits
        if "request" in answer_lower or "limit" in answer_lower:
            mistakes.append("mentions resource management")

        accuracy_score = 1.0 - (len(mistakes) * 0.1)  # Penalize each mistake
        accuracy_score = max(accuracy_score, 0.0)

        return {
            "accuracy_score": accuracy_score,
            "mentions": mistakes,
        }

    def evaluate_answer(
        self, answer: str, question: str, retrieved_documents: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Comprehensive answer evaluation

        Args:
            answer: Generated answer
            question: Original question
            retrieved_documents: List of retrieved documents (optional)

        Returns:
            Dict with overall quality score and component scores
        """
        retrieved_documents = retrieved_documents or []

        # Evaluate different aspects
        length_metrics = self.evaluate_answer_length(answer)
        content_metrics = self.evaluate_answer_content(answer, question)
        relevance_metrics = self.evaluate_relevance(answer, retrieved_documents)
        accuracy_metrics = self.evaluate_technical_accuracy(answer)

        # Combine scores
        overall_score = (
            length_metrics["length_score"] * 0.2
            + content_metrics["structure_score"] * 0.2
            + content_metrics["completeness_score"] * 0.2
            + relevance_metrics["source_citation_score"] * 0.2
            + accuracy_metrics["accuracy_score"] * 0.2
        )

        return {
            "overall_score": float(overall_score),
            "length": length_metrics,
            "content": content_metrics,
            "relevance": relevance_metrics,
            "accuracy": accuracy_metrics,
        }

    def compare_answers(self, answer1: str, answer2: str, question: str) -> Dict:
        """
        Compare two answers

        Args:
            answer1: First answer
            answer2: Second answer
            question: Question

        Returns:
            Comparison dict
        """
        eval1 = self.evaluate_answer(answer1, question)
        eval2 = self.evaluate_answer(answer2, question)

        return {
            "question": question,
            "answer1_score": eval1["overall_score"],
            "answer2_score": eval2["overall_score"],
            "winner": "answer1" if eval1["overall_score"] > eval2["overall_score"] else "answer2",
            "difference": abs(eval1["overall_score"] - eval2["overall_score"]),
            "eval1": eval1,
            "eval2": eval2,
        }

    def evaluate_batch(
        self, answers: List[str], questions: List[str], retrieved_docs_list: List[List[Dict]] = None
    ) -> Dict:
        """
        Evaluate batch of answers

        Args:
            answers: List of generated answers
            questions: List of questions
            retrieved_docs_list: List of retrieved documents per answer (optional)

        Returns:
            Batch evaluation results
        """
        retrieved_docs_list = retrieved_docs_list or [[] for _ in answers]

        all_evaluations = []
        scores = []

        for answer, question, docs in zip(answers, questions, retrieved_docs_list):
            eval_result = self.evaluate_answer(answer, question, docs)
            all_evaluations.append(eval_result)
            scores.append(eval_result["overall_score"])

        avg_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if scores else 0.0
        min_score = np.min(scores) if scores else 0.0
        max_score = np.max(scores) if scores else 0.0

        return {
            "num_answers": len(answers),
            "average_score": float(avg_score),
            "std_dev": float(std_score),
            "min_score": float(min_score),
            "max_score": float(max_score),
            "evaluations": all_evaluations,
        }
