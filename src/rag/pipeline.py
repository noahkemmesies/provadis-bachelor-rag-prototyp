"""RAG Pipeline combining Retrieval and Generation"""

from typing import Dict, List, Optional, Tuple

from src.config import CONTEXT_MAX_CHARS, CONTEXT_WINDOW, TOP_K_RETRIEVAL
from src.rag.llm_integration import OllamaLLM
from src.rag.prompting import PromptTemplates, format_context
from src.retrieval.base import BaseRetriever
from src.utils.logger import logger


class RAGPipeline:
    """RAG Pipeline: Retrieval + Augmented + Generation"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        context_window: int = CONTEXT_WINDOW,
        context_max_chars: int = CONTEXT_MAX_CHARS,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
    ):
        """
        Initialize RAG Pipeline

        Args:
            retriever: Retriever instance (BM25 or Embedding)
            llm: OllamaLLM instance (created if None)
            context_window: Number of documents to use as context
            context_max_chars: Maximum context size in characters
            top_k_retrieval: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm = llm or OllamaLLM()
        self.context_window = min(context_window, top_k_retrieval)
        self.context_max_chars = context_max_chars
        self.top_k_retrieval = top_k_retrieval

        self.system_prompt = (
            "You are a helpful Kubernetes assistant. "
            "Answer questions about Kubernetes based on the provided documentation. "
            "Be concise and accurate."
        )

        logger.info(f"Initialized RAGPipeline with {retriever.__class__.__name__}")
        logger.info(f"  Context window: {self.context_window}")
        logger.info(f"  Context max chars: {self.context_max_chars}")

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents

        Args:
            query: Query string

        Returns:
            List of retrieved documents
        """
        try:
            docs = self.retriever.retrieve(query, top_k=self.top_k_retrieval)
            logger.debug(f"Retrieved {len(docs)} documents for query")
            return docs
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents as context

        Args:
            documents: List of document dicts

        Returns:
            Formatted context string
        """
        try:
            # Limit to context_window documents
            context_docs = documents[: self.context_window]

            # Format context
            context = format_context(context_docs)

            # Truncate if too long
            if len(context) > self.context_max_chars:
                context = context[: self.context_max_chars] + "..."
                logger.debug(f"Truncated context to {self.context_max_chars} chars")

            return context
        except Exception as e:
            logger.error(f"Context formatting error: {e}")
            return "No context available"

    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using LLM

        Args:
            query: Original query
            context: Retrieved context

        Returns:
            Generated answer
        """
        try:
            prompt = PromptTemplates.format_qa(context=context, question=query)
            answer = self.llm.generate(prompt, system_prompt=self.system_prompt)
            return answer
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating answer: {str(e)}"

    def run(self, query: str) -> Dict:
        """
        Run complete RAG pipeline

        Args:
            query: Input query

        Returns:
            Dict with query, context, and answer
        """
        logger.info(f"Processing query: {query[:60]}...")

        # Step 1: Retrieve
        retrieved_docs = self.retrieve(query)

        # Step 2: Format context
        context = self.format_context(retrieved_docs)

        # Step 3: Generate answer
        answer = self.generate(query, context)

        result = {
            "query": query,
            "retrieved_documents": [
                {
                    "filename": doc.get("filename"),
                    "score": doc.get("score"),
                    "rank": doc.get("rank"),
                }
                for doc in retrieved_docs
            ],
            "context": context,
            "answer": answer,
        }

        logger.debug(f"Generated answer: {answer[:100]}...")

        return result

    def run_batch(self, queries: List[str]) -> List[Dict]:
        """
        Run RAG pipeline on multiple queries

        Args:
            queries: List of queries

        Returns:
            List of results
        """
        logger.info(f"Processing {len(queries)} queries...")

        results = []
        for i, query in enumerate(queries, 1):
            try:
                result = self.run(query)
                results.append(result)

                if i % 5 == 0:
                    logger.info(f"  Processed {i}/{len(queries)} queries")
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "answer": f"Error: {str(e)}",
                })

        logger.info(f"Completed batch processing: {len(results)}/{len(queries)}")
        return results


class HybridRAGPipeline:
    """Hybrid RAG Pipeline combining multiple retrievers"""

    def __init__(self, retrievers: Dict[str, BaseRetriever], llm: Optional[OllamaLLM] = None):
        """
        Initialize Hybrid RAG Pipeline

        Args:
            retrievers: Dict mapping retriever names to retriever instances
            llm: OllamaLLM instance
        """
        self.retrievers = retrievers
        self.llm = llm or OllamaLLM()
        self.pipelines = {
            name: RAGPipeline(retriever, llm) for name, retriever in retrievers.items()
        }

        logger.info(f"Initialized HybridRAGPipeline with {len(retrievers)} retrievers")

    def run(self, query: str) -> Dict[str, Dict]:
        """
        Run query through all retrievers

        Args:
            query: Input query

        Returns:
            Dict mapping retriever name to results
        """
        results = {}

        for name, pipeline in self.pipelines.items():
            logger.debug(f"Running {name} pipeline...")
            results[name] = pipeline.run(query)

        return results

    def run_batch(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """
        Run batch through all retrievers

        Args:
            queries: List of queries

        Returns:
            Dict mapping retriever name to list of results
        """
        results = {}

        for name, pipeline in self.pipelines.items():
            logger.info(f"Running batch through {name} pipeline...")
            results[name] = pipeline.run_batch(queries)

        return results

    def compare_answers(self, query: str) -> Dict:
        """
        Compare answers from different retrievers

        Args:
            query: Input query

        Returns:
            Dict with comparison
        """
        results = self.run(query)

        comparison = {
            "query": query,
            "retrievers": {},
        }

        for name, result in results.items():
            comparison["retrievers"][name] = {
                "num_documents": len(result.get("retrieved_documents", [])),
                "answer_length": len(result.get("answer", "")),
                "answer": result.get("answer"),
            }

        return comparison
