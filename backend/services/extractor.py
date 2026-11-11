"""
Named Entity Recognition & Relationship Extraction Engine

Supports two backends:
1. **Gemini API** (primary) — Structured prompt requesting entities and triples via
   google-genai, constrained to the six canonical entity categories.
2. **Rule-based NLP fallback** — Lightweight heuristic extractor using regex patterns
   for proper nouns and co-occurrence relationships, used when no API key is configured.

Entity categories: PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, CONCEPT, EVENT
Triple format:    (Subject, Predicate, Object, Evidence, Confidence)
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from backend.config import settings

logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "CONCEPT", "EVENT"}

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine. Analyze the following text and extract:

1. **Entities** — Named concepts mentioned in the text. Each entity must have:
   - "name": The canonical name (proper casing)
   - "type": One of: PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, CONCEPT, EVENT

2. **Relationships** — Directed semantic relationships between extracted entities. Each relationship must have:
   - "source": Name of the source entity (must match an extracted entity name)
   - "target": Name of the target entity (must match an extracted entity name)
   - "type": A short uppercase verb phrase describing the relationship (e.g., USES, DEVELOPED_BY, PART_OF, LOCATED_IN, INFLUENCED, WORKS_AT, FOUNDED, DEPENDS_ON, RELATED_TO)
   - "evidence": The exact sentence or phrase from the text that supports this relationship
   - "confidence": A float between 0.0 and 1.0 indicating confidence

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{
  "entities": [
    {"name": "...", "type": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "...", "evidence": "...", "confidence": 0.95}
  ]
}

If the text has no meaningful entities or relationships, return: {"entities": [], "relationships": []}

TEXT:
\"\"\"
__TEXT__
\"\"\"
"""


@dataclass
class ExtractedEntity:
    """A single extracted entity."""
    name: str
    type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": self.aliases,
        }


@dataclass
class ExtractedRelationship:
    """A single extracted directed relationship triple."""
    source: str
    target: str
    type: str
    evidence: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    """Complete extraction output for one chunk."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "error": self.error,
        }


def _normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type to one of the valid categories."""
    upper = entity_type.strip().upper()
    # Map common variants
    type_map = {
        "TECH": "TECHNOLOGY", "TOOL": "TECHNOLOGY", "SOFTWARE": "TECHNOLOGY",
        "FRAMEWORK": "TECHNOLOGY", "LANGUAGE": "TECHNOLOGY", "ALGORITHM": "TECHNOLOGY",
        "ORG": "ORGANIZATION", "COMPANY": "ORGANIZATION", "INSTITUTION": "ORGANIZATION",
        "PLACE": "LOCATION", "CITY": "LOCATION", "COUNTRY": "LOCATION",
        "IDEA": "CONCEPT", "THEORY": "CONCEPT", "METHOD": "CONCEPT",
        "TECHNIQUE": "CONCEPT", "PROCESS": "CONCEPT",
    }
    normalized = type_map.get(upper, upper)
    if normalized in VALID_ENTITY_TYPES:
        return normalized
    return "CONCEPT"


def _normalize_rel_type(rel_type: str) -> str:
    """Normalize relationship type to uppercase snake_case verb phrase."""
    cleaned = rel_type.strip().upper().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r'[^A-Z0-9_]', '', cleaned)
    return cleaned or "RELATED_TO"


# =============================================================================
# Backend 1: Gemini API Extractor
# =============================================================================

def _extract_with_gemini(text: str) -> ExtractionResult:
    """Use Google Gemini API for structured entity and relationship extraction."""
    from google import genai

    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    prompt = EXTRACTION_PROMPT.replace("__TEXT__", text)

    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=prompt,
    )

    raw_text = response.text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in the response
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse Gemini response as JSON: %s", raw_text[:200])
                return ExtractionResult(error=f"JSON parse error: {e}")
        else:
            return ExtractionResult(error=f"No JSON found in response: {raw_text[:200]}")

    entities = []
    for e in data.get("entities", []):
        name = e.get("name", "").strip()
        if not name or len(name) < 2:
            continue
        entities.append(ExtractedEntity(
            name=name,
            type=_normalize_entity_type(e.get("type", "CONCEPT")),
            description=e.get("description", ""),
        ))

    # Build set of valid entity names for relationship validation
    entity_names = {e.name.lower() for e in entities}

    relationships = []
    for r in data.get("relationships", []):
        source = r.get("source", "").strip()
        target = r.get("target", "").strip()
        if not source or not target:
            continue
        # Validate that source and target are known entities
        if source.lower() not in entity_names or target.lower() not in entity_names:
            continue
        relationships.append(ExtractedRelationship(
            source=source,
            target=target,
            type=_normalize_rel_type(r.get("type", "RELATED_TO")),
            evidence=r.get("evidence", ""),
            confidence=min(1.0, max(0.0, float(r.get("confidence", 0.9)))),
        ))

    return ExtractionResult(entities=entities, relationships=relationships)


# =============================================================================
# Backend 2: Rule-based NLP Fallback Extractor
# =============================================================================

# Technology/concept keyword patterns
_TECH_KEYWORDS = {
    "transformer", "bert", "gpt", "llm", "neural network", "deep learning",
    "machine learning", "natural language processing", "nlp", "cnn", "rnn",
    "attention", "self-attention", "multi-head attention", "embedding",
    "vector database", "knowledge graph", "rag", "retrieval", "generation",
    "api", "database", "algorithm", "python", "javascript", "java",
    "tensorflow", "pytorch", "neo4j", "chromadb", "fastapi", "react",
    "docker", "kubernetes", "sql", "nosql", "graphql", "rest",
    "convolutional", "recurrent", "encoder", "decoder", "tokenization",
    "fine-tuning", "pre-training", "inference", "backpropagation",
    "gradient descent", "optimization", "softmax", "relu",
}

_ORG_KEYWORDS = {
    "google", "openai", "microsoft", "meta", "amazon", "apple", "nvidia",
    "anthropic", "deepmind", "hugging face", "huggingface", "facebook",
    "ibm", "intel", "tesla", "spacex", "stanford", "mit", "berkeley",
    "university", "institute", "laboratory", "lab", "corp", "inc",
}

_LOCATION_KEYWORDS = {
    "california", "new york", "london", "beijing", "tokyo", "paris",
    "berlin", "san francisco", "seattle", "boston", "toronto",
    "silicon valley", "china", "japan", "india", "europe", "usa",
}

_RELATION_PATTERNS = [
    (r'(\w[\w\s-]+?)\s+(?:is |was )?(?:developed|created|built|designed|made)\s+by\s+(\w[\w\s-]+)',
     "DEVELOPED_BY"),
    (r'(\w[\w\s-]+?)\s+(?:uses?|utilizes?|employs?|leverages?)\s+(\w[\w\s-]+)',
     "USES"),
    (r'(\w[\w\s-]+?)\s+(?:is |are )?(?:based on|built on|relies on)\s+(\w[\w\s-]+)',
     "BASED_ON"),
    (r'(\w[\w\s-]+?)\s+(?:is |are )?(?:a |an )?(?:part of|component of|module of)\s+(\w[\w\s-]+)',
     "PART_OF"),
    (r'(\w[\w\s-]+?)\s+(?:is |are )?(?:a |an )?(?:type of|kind of|form of|variant of)\s+(\w[\w\s-]+)',
     "TYPE_OF"),
    (r'(\w[\w\s-]+?)\s+(?:depends on|requires)\s+(\w[\w\s-]+)',
     "DEPENDS_ON"),
    (r'(\w[\w\s-]+?)\s+(?:introduced|proposed|presented)\s+(\w[\w\s-]+)',
     "INTRODUCED"),
]


def _extract_with_rules(text: str) -> ExtractionResult:
    """
    Lightweight rule-based NER and RE for when no LLM API key is available.
    Uses keyword dictionaries and regex patterns to extract entities and relationships.
    """
    text_lower = text.lower()
    entities: Dict[str, ExtractedEntity] = {}

    # 1. Find technology/concept mentions
    for kw in _TECH_KEYWORDS:
        if kw in text_lower:
            # Find the original-case version in text
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                name = match.group().strip()
                canonical = name.title() if len(name) > 3 else name.upper()
                if canonical not in entities:
                    entities[canonical] = ExtractedEntity(name=canonical, type="TECHNOLOGY")

    # 2. Find organization mentions
    for kw in _ORG_KEYWORDS:
        if kw in text_lower:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                name = match.group().strip()
                canonical = name.title()
                if canonical not in entities:
                    entities[canonical] = ExtractedEntity(name=canonical, type="ORGANIZATION")

    # 3. Find location mentions
    for kw in _LOCATION_KEYWORDS:
        if kw in text_lower:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                name = match.group().strip().title()
                if name not in entities:
                    entities[name] = ExtractedEntity(name=name, type="LOCATION")

    # 4. Extract capitalized multi-word proper nouns as potential entities
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        name = match.group().strip()
        if len(name) > 3 and name not in entities:
            # Classify by context
            entities[name] = ExtractedEntity(name=name, type="CONCEPT")

    # 5. Extract relationships via patterns
    entity_names_lower = {e.name.lower(): e.name for e in entities.values()}
    relationships: List[ExtractedRelationship] = []

    for pattern, rel_type in _RELATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            source_raw = match.group(1).strip()
            target_raw = match.group(2).strip()

            # Match to known entities (case-insensitive)
            source_name = entity_names_lower.get(source_raw.lower())
            target_name = entity_names_lower.get(target_raw.lower())

            if source_name and target_name and source_name != target_name:
                # Extract surrounding sentence as evidence
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                evidence = text[start:end].strip()

                relationships.append(ExtractedRelationship(
                    source=source_name,
                    target=target_name,
                    type=rel_type,
                    evidence=evidence,
                    confidence=0.7,
                ))

    # 6. Co-occurrence relationships: entities in same sentence are likely related
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in sentences:
        sent_entities = []
        for ename, entity in entities.items():
            if ename.lower() in sentence.lower():
                sent_entities.append(ename)

        # Create co-occurrence edges between entities in same sentence (max 3 pairs)
        for i in range(min(len(sent_entities), 4)):
            for j in range(i + 1, min(len(sent_entities), 4)):
                # Check if this pair already has a relationship
                pair_exists = any(
                    (r.source == sent_entities[i] and r.target == sent_entities[j]) or
                    (r.source == sent_entities[j] and r.target == sent_entities[i])
                    for r in relationships
                )
                if not pair_exists:
                    relationships.append(ExtractedRelationship(
                        source=sent_entities[i],
                        target=sent_entities[j],
                        type="RELATED_TO",
                        evidence=sentence.strip()[:150],
                        confidence=0.5,
                    ))

    return ExtractionResult(
        entities=list(entities.values()),
        relationships=relationships,
    )


# =============================================================================
# Public API
# =============================================================================

def extract_from_text(text: str, use_llm: bool = True) -> ExtractionResult:
    """
    Extract entities and relationships from a text chunk.

    Args:
        text: The chunk text to analyze.
        use_llm: Whether to attempt Gemini API extraction first.

    Returns:
        ExtractionResult with entities and relationships.
    """
    if not text or not text.strip() or len(text.strip()) < 20:
        return ExtractionResult()

    if use_llm and settings.GEMINI_API_KEY:
        try:
            result = _extract_with_gemini(text)
            if result.entities or result.relationships:
                logger.info(
                    "Gemini extracted %d entities, %d relationships.",
                    len(result.entities), len(result.relationships)
                )
                return result
            # If Gemini returned nothing, fall through to rules
            logger.info("Gemini returned empty extraction, falling back to rules.")
        except Exception as e:
            logger.warning("Gemini extraction failed: %s. Falling back to rule-based.", e)

    # Fallback to rule-based extraction
    result = _extract_with_rules(text)
    logger.info(
        "Rule-based extracted %d entities, %d relationships.",
        len(result.entities), len(result.relationships)
    )
    return result


def extract_from_chunks(
    chunks: List[Dict[str, Any]],
    document_id: str,
    use_llm: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract entities and relationships from a list of chunks.

    Returns:
        Tuple of (all_entities, all_relationships) as dicts ready for graph store ingestion.
        Each entity dict includes document_id. Each relationship dict includes
        document_id and chunk_id provenance.
    """
    seen_entities: Dict[str, Dict[str, Any]] = {}
    all_relationships: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = chunk.get("id") or chunk.get("chunk_id", "")
        content = chunk.get("content", "")

        result = extract_from_text(content, use_llm=use_llm)
        if result.error:
            logger.warning("Extraction error for chunk %s: %s", chunk_id, result.error)

        # Deduplicate entities across chunks
        for entity in result.entities:
            key = entity.name.upper().strip()
            if key in seen_entities:
                # Merge document_ids
                existing = seen_entities[key]
                if document_id not in existing.get("document_ids", []):
                    existing.setdefault("document_ids", []).append(document_id)
            else:
                seen_entities[key] = {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "aliases": entity.aliases,
                    "document_id": document_id,
                    "document_ids": [document_id],
                }

        # Add provenance to relationships
        for rel in result.relationships:
            all_relationships.append({
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "evidence": rel.evidence,
                "confidence": rel.confidence,
                "document_id": document_id,
                "chunk_id": chunk_id,
            })

    return list(seen_entities.values()), all_relationships

