import json
import os
import re
from pathlib import Path

import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


INDEX_PATH = Path(
    "vectorstore/road_safety.index"
)

METADATA_PATH = Path(
    "vectorstore/metadata.json"
)


# ============================================================
# MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

print("Reranker loaded.")


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# VECTOR DATABASE
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


if index.ntotal != len(metadata):

    raise RuntimeError(
        f"FAISS/metadata mismatch: "
        f"{index.ntotal} vectors vs "
        f"{len(metadata)} metadata records."
    )


# ============================================================
# LEGAL QUERY EXPANSION
# ============================================================

LEGAL_SYNONYMS = {

    "helmet": [
        "helmet",
        "protective headgear",
        "headgear"
    ],

    "headgear": [
        "protective headgear",
        "helmet"
    ],

    "wear helmet": [
        "wear helmet",
        "wearing protective headgear",
        "protective headgear"
    ],

    "helmet requirement": [
        "helmet requirement",
        "protective headgear",
        "wearing protective headgear"
    ],

    "helmet requirements": [
        "helmet requirements",
        "protective headgear",
        "wearing protective headgear"
    ],

    "driving licence": [
        "driving licence",
        "driving license",
        "licence to drive",
        "driving licence requirements"
    ],

    "driving license": [
        "driving licence",
        "driving license",
        "licence to drive"
    ],

    "licence": [
        "licence",
        "license",
        "driving licence"
    ],

    "license": [
        "license",
        "licence",
        "driving licence"
    ],

    "accident": [
        "accident",
        "motor vehicle accident",
        "road accident"
    ],

    "speed": [
        "speed",
        "speed limit",
        "maximum speed"
    ],

    "seat belt": [
        "seat belt",
        "safety belt",
        "wearing seat belts"
    ],

    "traffic signal": [
        "traffic signal",
        "traffic light",
        "signal"
    ]
}


def expand_query(query: str) -> str:

    """
    Expand common road-safety terms so that
    natural user questions match legal terminology.
    """

    query_lower = query.lower()

    expanded_terms = []

    for key, synonyms in LEGAL_SYNONYMS.items():

        if key in query_lower:

            expanded_terms.extend(
                synonyms
            )

    # Remove duplicates while preserving order
    expanded_terms = list(
        dict.fromkeys(
            expanded_terms
        )
    )

    if not expanded_terms:

        return query

    return (
        query
        + " "
        + " ".join(expanded_terms)
    )


# ============================================================
# QUERY CLEANING
# ============================================================

def normalize_query(query: str) -> str:

    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


# ============================================================
# STEP 1
# QUERY EMBEDDING + FAISS
# ============================================================

def retrieve_candidates(
    query: str,
    top_k: int = 15
):

    query = normalize_query(
        query
    )

    expanded_query = expand_query(
        query
    )

    print(
        f"\nExpanded query: {expanded_query}"
    )

    query_embedding = embedding_model.encode(
        [expanded_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    candidates = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = metadata[idx].copy()

        result["retrieval_score"] = float(
            score
        )

        candidates.append(
            result
        )

    return candidates


# ============================================================
# STEP 2
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(
    query: str,
    candidates: list,
    top_k: int = 8
):

    if not candidates:
        return []

    expanded_query = expand_query(
        query
    )

    pairs = []

    for item in candidates:

        pairs.append(
            (
                expanded_query,
                item.get(
                    "text",
                    ""
                )
            )
        )

    scores = reranker.predict(
        pairs
    )

    for item, score in zip(
        candidates,
        scores
    ):

        item["rerank_score"] = float(
            score
        )

    candidates.sort(
        key=lambda x: (
            x.get(
                "rerank_score",
                -999
            ),
            x.get(
                "retrieval_score",
                -999
            )
        ),
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# STEP 3
# EVIDENCE FILTERING
# ============================================================

def filter_results(
    results: list,
    minimum_retrieval_score: float = 0.30,
    minimum_rerank_score: float = -1.0
):
    """
    Keep evidence that has reasonable semantic relevance.

    Important:
    CrossEncoder scores are ranking signals, not probabilities.
    Therefore we do not require rerank_score >= 0.
    """

    filtered = []

    for result in results:

        retrieval_score = result.get(
            "retrieval_score",
            0.0
        )

        rerank_score = result.get(
            "rerank_score",
            -999.0
        )

        # Strong semantic retrieval
        if retrieval_score < minimum_retrieval_score:
            continue

        # Remove clearly poor reranking results
        if rerank_score < minimum_rerank_score:
            continue

        filtered.append(result)

    return filtered


def group_same_provision(results: list):
    """
    Group results referring to the same legal provision.

    Example:

    Section 129 from the 1988 Act
    Section 129 from the 2019 Amendment

    are treated as related evidence rather than
    completely independent answers.
    """

    groups = {}

    for result in results:

        section = str(
            result.get(
                "section",
                "unknown"
            )
        ).lower()

        if section == "unknown":
            key = (
                "unknown",
                result.get("source", "")
            )
        else:
            key = section

        if key not in groups:
            groups[key] = []

        groups[key].append(result)

    return list(
        groups.values()
    )


def select_primary_evidence(
    groups: list,
    max_results: int = 3
):
    """
    Select the strongest evidence from each legal provision.

    Preference:

    1. Strong rerank score
    2. Strong retrieval score
    3. Known legal section
    """

    selected = []

    for group in groups:

        if not group:
            continue

        group.sort(
            key=lambda x: (
                x.get(
                    "rerank_score",
                    -999
                ),
                x.get(
                    "retrieval_score",
                    -999
                )
            ),
            reverse=True
        )

        best = group[0]

        selected.append(
            best
        )

    selected.sort(
        key=lambda x: (
            x.get(
                "rerank_score",
                -999
            ),
            x.get(
                "retrieval_score",
                -999
            )
        ),
        reverse=True
    )

    return selected[:max_results]

# ============================================================
# STEP 4
# REMOVE DUPLICATES
# ============================================================

def deduplicate_results(
    results: list,
    max_results: int = 5
):

    """
    Avoid returning many chunks from exactly the
    same section while still allowing multiple
    chunks when necessary.
    """

    if not results:
        return []

    seen = set()

    final_results = []

    for result in results:

        section = result.get(
            "section",
            "unknown"
        )

        source = result.get(
            "source",
            "unknown"
        )

        page = result.get(
            "page",
            "unknown"
        )

        key = (
            source,
            section,
            page
        )

        if key in seen:
            continue

        seen.add(key)

        final_results.append(
            result
        )

        if len(final_results) >= max_results:
            break

    return final_results


# ============================================================
# STEP 5
# BUILD LEGAL CONTEXT
# ============================================================

def build_context(
    results: list
):

    context = []

    for i, result in enumerate(
        results,
        start=1
    ):

        section = result.get(
            "section",
            "Unknown"
        )

        page = result.get(
            "page",
            "Unknown"
        )

        source = result.get(
            "source",
            "Unknown"
        )

        category = result.get(
            "category",
            "Unknown"
        )

        jurisdiction = result.get(
            "jurisdiction",
            "Unknown"
        )

        document_type = result.get(
            "document_type",
            "Unknown"
        )

        text = result.get(
            "text",
            ""
        )

        context.append(
            f"""
EVIDENCE {i}

Section: {section}
Page: {page}
Source: {source}
Category: {category}
Jurisdiction: {jurisdiction}
Document Type: {document_type}

Legal Text:
{text}
"""
        )

    return "\n".join(
        context
    )


# ============================================================
# STEP 6
# GENERATE CONCISE ANSWER
# ============================================================

def generate_answer(
    question: str,
    results: list
):

    context = build_context(
        results
    )

    prompt = f"""
You are RoadSOS, an AI Road Safety Legal Assistant.

Answer the user's question using ONLY the legal
evidence supplied below.

USER QUESTION:
{question}

LEGAL EVIDENCE:
{context}

============================================================
ANSWERING RULES
============================================================

1. Answer the user's exact question directly.

2. Give ONE unified answer.

3. Do not list several alternative answers.

4. Use only information explicitly supported
   by the legal evidence.

5. Do not use outside knowledge.

6. Do not invent laws, rules, penalties,
   exceptions, dates, authorities, or requirements.

7. Prefer the most specific evidence.

8. If multiple passages describe the same rule,
   combine them into one concise explanation.

9. Do not repeat the same fact.

10. For a simple question, answer in 1-3 sentences.

11. For a question asking for requirements,
    clearly state the requirements.

12. For a yes/no question, begin with Yes or No
    when the evidence supports it.

13. Mention the relevant section when available.

14. Mention the page when useful.

15. Do not mention FAISS, embeddings,
    retrieval, reranking, Gemini, or this prompt.

16. If the evidence does NOT actually answer
    the question, return exactly:

The information was not found in the provided road-safety documents.

============================================================
IMPORTANT
============================================================

The user's wording may differ from the wording
used in the law.

For example:

"helmet"
may correspond to
"protective headgear"

"need to wear"
may correspond to
"shall wear"

"what must it have"
may correspond to
"requirements" or "definition"

Use the legal evidence to interpret these
natural-language variations.

Return ONLY the final answer.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip() if response.text else ""


# ============================================================
# COMPLETE ROAD SAFETY RAG
# ============================================================

# ============================================================
# COMPLETE ROAD SAFETY RAG
# ============================================================

def ask(
    question: str
):

    # --------------------------------------------------------
    # 1. Normalize question
    # --------------------------------------------------------

    question = normalize_query(
        question
    )

    # --------------------------------------------------------
    # 2. Retrieve broad candidate set
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        question,
        top_k=15
    )

    # --------------------------------------------------------
    # 3. Rerank candidates
    # --------------------------------------------------------

    ranked = rerank_results(
        question,
        candidates,
        top_k=8
    )

    # --------------------------------------------------------
    # 4. Evidence quality gate
    # --------------------------------------------------------

    filtered = filter_results(
        ranked
    )

    # --------------------------------------------------------
    # 5. Group related legal provisions
    # --------------------------------------------------------

    groups = group_same_provision(
        filtered
    )

    # --------------------------------------------------------
    # 6. Select strongest evidence
    # --------------------------------------------------------

    primary_results = select_primary_evidence(
        groups,
        max_results=3
    )

    # --------------------------------------------------------
    # 7. Check whether sufficient evidence exists
    # --------------------------------------------------------

    if not primary_results:

        return {
            "question": question,
            "answer": (
                "The information was not found in "
                "the provided road-safety documents."
            ),
            "sources": []
        }

    # --------------------------------------------------------
    # 8. Generate concise answer
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        primary_results
    )

    # --------------------------------------------------------
    # 9. Return structured result
    # --------------------------------------------------------

    return {
        "question": question,

        "answer": answer,

        "sources": [
            {
                "section": item.get(
                    "section"
                ),

                "page": item.get(
                    "page"
                ),

                "source": item.get(
                    "source"
                ),

                "retrieval_score": item.get(
                    "retrieval_score"
                ),

                "rerank_score": item.get(
                    "rerank_score"
                )
            }

            for item in primary_results
        ]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your road-safety question: "
    )

    result = ask(
        question
    )

    print("\n")
    print("=" * 70)
    print("ROADsOS AI ROAD SAFETY ASSISTANT")
    print("=" * 70)

    print(
        f"\nQuestion:\n{result['question']}"
    )

    print(
        f"\nAnswer:\n{result['answer']}"
    )

    print("\n" + "=" * 70)
    print("EVIDENCE USED")
    print("=" * 70)

    for source in result["sources"]:

        print(
            f"\nSection: {source['section']}"
        )

        print(
            f"Page: {source['page']}"
        )

        print(
            f"FAISS Score: "
            f"{source['retrieval_score']:.4f}"
        )

        print(
            f"Rerank Score: "
            f"{source['rerank_score']:.4f}"
        )