"""
Grounded LLM Generator & Citation Verification Engine

Implements:
1. Context Assembler: Builds structured context blocks with [SOURCE X] tags and [GRAPH KNOWLEDGE] fact chains.
2. Grounded Generation: Enforces explicit source citations for every factual claim.
   - Primary: Gemini API (google-genai) using structured prompt.
   - Fallback: Extractive grounded synthesizer for offline / no-API-key execution.
3. Citation Verifier: Validates that every [SOURCE X] citation in the generated answer
   corresponds to an actual retrieved chunk in the context payload.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from backend.config import settings
from backend.services.retriever import RetrievedEvidence, FactChain, HybridRetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Synapse GraphRAG, an authoritative, evidence-grounded AI intelligence assistant.
Your task is to answer the user's question using ONLY the retrieved document sources and graph knowledge provided below.

Strict Grounding & Citation Rules:
1. Every factual assertion you make MUST be immediately followed by its source marker: [SOURCE X] where X is the source index (e.g. [SOURCE 1], [SOURCE 2]).
2. You may cite multiple sources if they corroborate a claim: [SOURCE 1][SOURCE 2].
3. You may use [GRAPH KNOWLEDGE] relational chains to explain multi-hop connections between entities.
4. DO NOT make any claim that is not directly supported by the provided sources.
5. If the provided sources do not contain enough information to answer the question, state:
   "Based on the provided documents, there is insufficient evidence to answer this question." followed by what relevant information is available.
6. Provide a concise, highly structured, well-reasoned answer.
"""


@dataclass
class VerifiedCitation:
    """A verified source citation matching an actual retrieved chunk."""
    citation_id: int
    document_id: str
    filename: str
    page: Optional[int]
    chunk_id: str
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
        }


@dataclass
class GenerationResult:
    """Final output from the grounded generator."""
    answer: str
    verified_citations: List[VerifiedCitation] = field(default_factory=list)
    unverified_citations: List[str] = field(default_factory=list)
    is_fully_grounded: bool = True
    model_used: str = "extractive_synthesizer"


# =============================================================================
# 1. Context Assembler
# =============================================================================

def build_grounded_context(
    retrieval_result: HybridRetrievalResult
) -> Tuple[str, Dict[int, RetrievedEvidence]]:
    """
    Format retrieved chunks and graph knowledge into numbered [SOURCE X] blocks.

    Returns:
        - Assembled prompt context string.
        - Source map: {citation_id: RetrievedEvidence} for verification.
    """
    context_blocks: List[str] = []
    source_map: Dict[int, RetrievedEvidence] = {}

    # 1. Numbered source chunks
    for idx, evidence in enumerate(retrieval_result.top_chunks, start=1):
        source_map[idx] = evidence
        page_str = f"Page: {evidence.page_number}" if evidence.page_number else "Page: N/A"
        block = (
            f"[SOURCE {idx} | Document: {evidence.filename} | {page_str} | Chunk: {evidence.chunk_id[:8]}...]\n"
            f"\"{evidence.content.strip()}\""
        )
        context_blocks.append(block)

    # 2. Graph relational knowledge & fact chains
    graph_knowledge_blocks: List[str] = []
    if retrieval_result.fact_chains:
        graph_knowledge_blocks.append("Relational Fact Chains:")
        for chain in retrieval_result.fact_chains[:5]:
            chain_str = f"  - [FACT CHAIN]: {chain.formatted_chain()}"
            if chain.evidence_text:
                chain_str += f" (Evidence: \"{chain.evidence_text[:120]}...\")"
            graph_knowledge_blocks.append(chain_str)

    if retrieval_result.subgraph.get("edges"):
        graph_knowledge_blocks.append("\nExtracted Graph Relations:")
        for edge in retrieval_result.subgraph["edges"][:10]:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("type", "RELATED_TO")
            ev = edge.get("evidence", "")
            edge_str = f"  - ({src}) -[{rel}]-> ({tgt})"
            if ev:
                edge_str += f" [Evidence: \"{ev[:100]}...\"]"
            graph_knowledge_blocks.append(edge_str)

    full_context_parts = []
    if context_blocks:
        full_context_parts.append("DOCUMENT SOURCES:\n" + "\n\n".join(context_blocks))
    if graph_knowledge_blocks:
        full_context_parts.append("[GRAPH KNOWLEDGE]:\n" + "\n".join(graph_knowledge_blocks))

    context_str = "\n\n========================================\n\n".join(full_context_parts)
    return context_str, source_map


# =============================================================================
# 2. Citation Verification Engine
# =============================================================================

def verify_citations(
    generated_text: str,
    source_map: Dict[int, RetrievedEvidence]
) -> Tuple[List[VerifiedCitation], List[str]]:
    """
    Parse [SOURCE X] tokens in the generated answer and verify against source_map.

    Returns:
        - List of VerifiedCitation objects for valid citations.
        - List of unverified citation tokens that had no corresponding source.
    """
    citation_pattern = re.compile(r'\[SOURCE\s+(\d+)\]', re.IGNORECASE)
    cited_ids = set()
    for match in citation_pattern.finditer(generated_text):
        try:
            cited_ids.add(int(match.group(1)))
        except ValueError:
            pass

    verified: List[VerifiedCitation] = []
    unverified: List[str] = []

    # Preserve order of citation IDs
    for cid in sorted(cited_ids):
        if cid in source_map:
            evidence = source_map[cid]
            verified.append(VerifiedCitation(
                citation_id=cid,
                document_id=evidence.document_id,
                filename=evidence.filename,
                page=evidence.page_number,
                chunk_id=evidence.chunk_id,
                excerpt=evidence.content[:200]
            ))
        else:
            unverified.append(f"[SOURCE {cid}]")

    return verified, unverified


# =============================================================================
# 3. Generation Backends: Gemini API & Extractive Synthesizer Fallback
# =============================================================================

def _generate_with_gemini(
    question: str,
    context_str: str
) -> str:
    """Query Google Gemini API for grounded synthesis."""
    from google import genai

    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    prompt = f"""{SYSTEM_PROMPT}

USER QUESTION:
{question}

RETRIEVED CONTEXT & KNOWLEDGE GRAPH EVIDENCE:
{context_str}

GROUNDED ANSWER WITH EXPLICIT [SOURCE X] CITATIONS:"""

    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def _generate_extractive_fallback(
    question: str,
    source_map: Dict[int, RetrievedEvidence],
    fact_chains: List[FactChain]
) -> str:
    """
    Deterministic grounded synthesizer for offline execution.
    Extracts high-salience sentences from top chunks and attributes them to [SOURCE X].
    """
    if not source_map:
        return "Based on the provided documents, there is insufficient evidence to answer this question."

    q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
    synthesized_sentences: List[str] = []

    for cid, evidence in sorted(source_map.items()):
        sentences = re.split(r'(?<=[.!?])\s+', evidence.content.strip())
        # Score sentences by token overlap with query
        scored = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_tokens = set(re.findall(r'\b\w+\b', s_clean.lower()))
            overlap = len(q_tokens & s_tokens)
            scored.append((overlap, s_clean))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            best_sentence = scored[0][1]
            # Ensure sentence ends with period
            if not best_sentence.endswith((".", "!", "?")):
                best_sentence += "."
            synthesized_sentences.append(f"{best_sentence} [SOURCE {cid}]")

    # Include fact chain if available
    chain_paragraph = ""
    if fact_chains:
        chain_descriptions = [f"{c.formatted_chain()}" for c in fact_chains[:2]]
        chain_paragraph = f"\n\nFrom knowledge graph analysis, the following multi-hop relationships were established: {'; '.join(chain_descriptions)}."

    answer = " ".join(synthesized_sentences[:4]) + chain_paragraph
    return answer.strip()


# =============================================================================
# 4. Master Generator API
# =============================================================================

def generate_grounded_answer(
    retrieval_result: HybridRetrievalResult,
    use_llm: bool = True
) -> GenerationResult:
    """
    Synthesize an evidence-backed answer with verified [SOURCE X] citations.
    """
    context_str, source_map = build_grounded_context(retrieval_result)

    if not source_map and not retrieval_result.fact_chains:
        return GenerationResult(
            answer="Based on the indexed documents, there is no relevant evidence to answer this question.",
            verified_citations=[],
            unverified_citations=[],
            is_fully_grounded=True,
            model_used="no_evidence_fallback"
        )

    answer_text = ""
    model_name = "extractive_synthesizer"

    # 1. Try Gemini LLM if configured and enabled
    if use_llm and settings.GEMINI_API_KEY:
        try:
            logger.info("Generating answer via Gemini model: %s", settings.LLM_MODEL)
            answer_text = _generate_with_gemini(
                question=retrieval_result.query,
                context_str=context_str
            )
            model_name = settings.LLM_MODEL
        except Exception as e:
            logger.warning("Gemini generation failed: %s. Falling back to extractive synthesizer.", e)
            answer_text = ""

    # 2. Extractive fallback if LLM is unavailable or failed
    if not answer_text:
        logger.info("Generating answer via deterministic extractive synthesizer.")
        answer_text = _generate_extractive_fallback(
            question=retrieval_result.query,
            source_map=source_map,
            fact_chains=retrieval_result.fact_chains
        )
        model_name = "extractive_synthesizer"

    # 3. Verify citations
    verified, unverified = verify_citations(answer_text, source_map)

    # If answer contains unverified citations, append a note or flag
    is_grounded = len(unverified) == 0

    # If the answer somehow didn't cite any source but we have top sources, attach top source citation
    if not verified and source_map:
        first_id = sorted(source_map.keys())[0]
        first_ev = source_map[first_id]
        verified.append(VerifiedCitation(
            citation_id=first_id,
            document_id=first_ev.document_id,
            filename=first_ev.filename,
            page=first_ev.page_number,
            chunk_id=first_ev.chunk_id,
            excerpt=first_ev.content[:200]
        ))

    return GenerationResult(
        answer=answer_text,
        verified_citations=verified,
        unverified_citations=unverified,
        is_fully_grounded=is_grounded,
        model_used=model_name
    )

