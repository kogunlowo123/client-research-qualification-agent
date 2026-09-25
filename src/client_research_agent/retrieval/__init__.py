"""Chunking, embedding, indexing and advanced retrieval (hybrid, multi-query, GraphRAG, CRAG, Self-RAG)."""

from client_research_agent.retrieval.base import ChunkRetriever
from client_research_agent.retrieval.bm25 import BM25Retriever
from client_research_agent.retrieval.chunking import (
    Chunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    make_chunk_id,
)
from client_research_agent.retrieval.compression import ContextCompressor
from client_research_agent.retrieval.crag import (
    CorrectiveRetriever,
    CragAction,
    CragResult,
    RelevanceGrader,
    RetrievalVerdict,
)
from client_research_agent.retrieval.dense import DenseRetriever
from client_research_agent.retrieval.embeddings import (
    CachingEmbeddingClient,
    HashingEmbeddingClient,
    cosine_similarity,
    embed_in_batches,
)
from client_research_agent.retrieval.enrichment import MetadataEnricher, extract_candidate_entities
from client_research_agent.retrieval.graph import Community, EntityGraph, GraphRetriever
from client_research_agent.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from client_research_agent.retrieval.indexing import IndexingPipeline, IndexingResult
from client_research_agent.retrieval.multi_query import MultiQueryResult, MultiQueryRetriever
from client_research_agent.retrieval.parent_child import ParentExpander
from client_research_agent.retrieval.pipeline import RetrievalOutcome, RetrievalPipeline, Retriever
from client_research_agent.retrieval.query_rewriting import QueryRewrite, QueryRewriter
from client_research_agent.retrieval.reranking import (
    LexicalReranker,
    LLMReranker,
    Reranker,
    maximal_marginal_relevance,
)
from client_research_agent.retrieval.self_rag import (
    AnswerCritique,
    SelfRagCritic,
    SelfRagReport,
    SupportLevel,
)
from client_research_agent.retrieval.tokenization import Tokenizer, count_tokens, get_tokenizer

__all__ = [
    "AnswerCritique",
    "BM25Retriever",
    "CachingEmbeddingClient",
    "ChunkRetriever",
    "Chunker",
    "Community",
    "ContextCompressor",
    "CorrectiveRetriever",
    "CragAction",
    "CragResult",
    "DenseRetriever",
    "EntityGraph",
    "GraphRetriever",
    "HashingEmbeddingClient",
    "HybridRetriever",
    "IndexingPipeline",
    "IndexingResult",
    "LLMReranker",
    "LexicalReranker",
    "MetadataEnricher",
    "MultiQueryResult",
    "MultiQueryRetriever",
    "ParentChildChunker",
    "ParentExpander",
    "QueryRewrite",
    "QueryRewriter",
    "RecursiveChunker",
    "RelevanceGrader",
    "Reranker",
    "RetrievalOutcome",
    "RetrievalPipeline",
    "RetrievalVerdict",
    "Retriever",
    "SelfRagCritic",
    "SelfRagReport",
    "SemanticChunker",
    "SupportLevel",
    "Tokenizer",
    "cosine_similarity",
    "count_tokens",
    "embed_in_batches",
    "extract_candidate_entities",
    "get_tokenizer",
    "make_chunk_id",
    "maximal_marginal_relevance",
    "reciprocal_rank_fusion",
]
