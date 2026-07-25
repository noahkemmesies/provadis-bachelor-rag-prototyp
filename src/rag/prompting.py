"""Prompt templates for RAG pipeline"""

from string import Template


class PromptTemplates:
    """Collection of prompt templates for RAG"""

    # Basic QA template
    QA_TEMPLATE = Template("""You are a helpful Kubernetes assistant. Answer the following question based on the provided context.

Context:
$context

Question: $question

Answer:""")

    # Detailed QA template with instruction
    DETAILED_QA_TEMPLATE = Template("""You are an expert Kubernetes assistant. Your task is to provide accurate and helpful answers about Kubernetes concepts and operations.

Instructions:
- Answer based on the provided context
- If the context doesn't contain relevant information, say so
- Provide clear and concise explanations
- Use technical terminology where appropriate

Context:
$context

Question: $question

Answer:""")

    # Summarization template
    SUMMARIZATION_TEMPLATE = Template("""Summarize the following Kubernetes documentation in 2-3 sentences:

$context

Summary:""")

    # Comparison template
    COMPARISON_TEMPLATE = Template("""Based on the provided context, compare and contrast the following:

$context

$question

Comparison:""")

    # Troubleshooting template
    TROUBLESHOOTING_TEMPLATE = Template("""You are a Kubernetes troubleshooting expert. Based on the provided documentation and the issue described, provide a solution:

Context:
$context

Issue: $question

Solution:""")

    @staticmethod
    def format_qa(context: str, question: str) -> str:
        """Format QA template"""
        return PromptTemplates.QA_TEMPLATE.substitute(context=context, question=question)

    @staticmethod
    def format_detailed_qa(context: str, question: str) -> str:
        """Format detailed QA template"""
        return PromptTemplates.DETAILED_QA_TEMPLATE.substitute(
            context=context, question=question
        )

    @staticmethod
    def format_summarization(context: str) -> str:
        """Format summarization template"""
        return PromptTemplates.SUMMARIZATION_TEMPLATE.substitute(context=context)

    @staticmethod
    def format_comparison(context: str, question: str) -> str:
        """Format comparison template"""
        return PromptTemplates.COMPARISON_TEMPLATE.substitute(
            context=context, question=question
        )

    @staticmethod
    def format_troubleshooting(context: str, question: str) -> str:
        """Format troubleshooting template"""
        return PromptTemplates.TROUBLESHOOTING_TEMPLATE.substitute(
            context=context, question=question
        )


def format_context(retrieved_docs: list) -> str:
    """
    Format retrieved documents as context string

    Args:
        retrieved_docs: List of retrieved document dictionaries

    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.get("content", "")
        filename = doc.get("filename", "unknown")
        score = doc.get("score", 0)

        # Limit content length
        if len(content) > 500:
            content = content[:500] + "..."

        context_parts.append(
            f"[Document {i}: {filename} (relevance: {score:.3f})]\n{content}"
        )

    return "\n\n".join(context_parts)
