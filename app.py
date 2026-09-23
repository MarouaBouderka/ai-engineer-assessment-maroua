import json
import os
import requests
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPERHERO_API_TOKEN = os.getenv("SUPERHERO_API_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

if not SUPERHERO_API_TOKEN:
    raise ValueError("SUPERHERO_API_TOKEN is not set.")

client = Groq(api_key=GROQ_API_KEY)

SUPERHERO_API_BASE_URL = "https://superheroapi.com/api"


def search_superhero(name: str) -> dict:
    """
    Search for a superhero using the Superhero API.
    """

    url = f"{SUPERHERO_API_BASE_URL}/{SUPERHERO_API_TOKEN}/search/{name}"

    try:
        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        raise RuntimeError(
            f"Superhero API request failed: {str(e)}"
        )
        
        
text_documents = {
    "rag.txt": """
Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation. Instead of relying only
on the knowledge stored in a language model, RAG retrieves relevant
documents from an external knowledge source and provides them to the
language model as context.

A typical RAG system has two main stages. First, a retriever searches
a collection of documents and finds the most relevant passages.
Second, a language model uses the retrieved passages to generate an
answer.

RAG is useful when the information comes from a private, changing,
or domain-specific knowledge base.
""",

    "llm.txt": """
Large Language Models (LLMs) are machine learning models trained on
large collections of text. They can perform tasks such as text
generation, summarization, question answering, and information
extraction.

LLMs generate responses based on patterns learned during training.
They can also be provided with external context at inference time,
for example through Retrieval-Augmented Generation.
""",

    "machine_learning.txt": """
Machine learning is a branch of artificial intelligence in which
models learn patterns from data.

Supervised learning uses labeled examples to train a model.
Unsupervised learning works with data without explicit labels.
Common supervised learning tasks include classification and
regression.

A machine learning pipeline commonly includes data preparation,
feature processing, model training, validation, and evaluation.
"""
}
    
def chunk_text(text: str, chunk_size: int = 500):
    """
    Split text into smaller chunks.
    """

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


text_chunks = []

for document_name, document_text in text_documents.items():

    chunks = chunk_text(document_text)

    for chunk in chunks:
        text_chunks.append({
            "document": document_name,
            "content": chunk
        })
chunk_contents = [
    chunk["content"]
    for chunk in text_chunks
]

vectorizer = TfidfVectorizer(
    stop_words="english"
)

text_matrix = vectorizer.fit_transform(chunk_contents)

def retrieve_text(
    question: str,
    top_k: int = 3,
    threshold: float = 0.1
):
    """
    Retrieve the most relevant text chunks for a question.
    """

    query_vector = vectorizer.transform([question])

    similarity_scores = cosine_similarity(
        query_vector,
        text_matrix
    )[0]

    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        score = float(similarity_scores[index])

        if score >= threshold:

            results.append({
                "document": text_chunks[index]["document"],
                "content": text_chunks[index]["content"],
                "score": score
            })

    return results

class Route(str, Enum):
    TEXT = "text"
    SUPERHERO = "superhero"
    BOTH = "both"
    UNKNOWN = "unknown"


class RoutingDecision(BaseModel):
    route: Route
    superhero_name: Optional[str] = None
    reason: str = Field(
        min_length=1,
        max_length=300
    )
ROUTER_SYSTEM_PROMPT = """
You are a routing component for a question-answering system.

The system has two information sources:

1. TEXT:
   A local text dataset containing domain-specific information.

2. SUPERHERO:
   The Superhero API, which contains information about superhero characters.

Your job is to determine which source or sources are needed to answer the user's question.

Choose exactly one route:

- "text": The question can be answered using the text dataset.
- "superhero": The question requires superhero information.
- "both": The question requires information from both sources.
- "unknown": Neither source is sufficient or the intent is unclear.

If the question refers to a superhero, identify the superhero name when possible.

Return ONLY valid JSON with this structure:

{
    "route": "text | superhero | both | unknown",
    "superhero_name": "string or null",
    "reason": "short explanation"
}
"""


def llm_route(question: str) -> RoutingDecision:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={"type": "json_object"}
    )

    raw_output = response.choices[0].message.content

    decision = RoutingDecision.model_validate(
        json.loads(raw_output)
    )

    return decision

def verify_route(
    question: str,
    decision: RoutingDecision
) -> RoutingDecision:

    # Verify TEXT source
    if decision.route == Route.TEXT:

        text_results = retrieve_text(question)

        if not text_results:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="The text route was proposed, but no relevant text was found."
            )

        return RoutingDecision(
            route=Route.TEXT,
            superhero_name=None,
            reason="The proposed text route was verified by the text retriever."
        )

    # Verify SUPERHERO source
    if decision.route == Route.SUPERHERO:

        if not decision.superhero_name:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="The superhero route was proposed, but no superhero name was identified."
            )

        try:
            superhero_data = search_superhero(
                decision.superhero_name
            )

            if superhero_data.get("response") != "success":
                return RoutingDecision(
                    route=Route.UNKNOWN,
                    superhero_name=decision.superhero_name,
                    reason="The superhero could not be verified using the Superhero API."
                )

            return RoutingDecision(
                route=Route.SUPERHERO,
                superhero_name=decision.superhero_name,
                reason="The proposed superhero route was verified using the Superhero API."
            )

        except RuntimeError:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The Superhero API could not be reached."
            )

    # Verify BOTH sources
    if decision.route == Route.BOTH:

        if not decision.superhero_name:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="The both route was proposed, but no superhero name was identified."
            )

        # Verify text source
        text_results = retrieve_text(question)

        if not text_results:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The both route was proposed, but no relevant text was found."
            )

        # Verify superhero source
        try:
            superhero_data = search_superhero(
                decision.superhero_name
            )

            if superhero_data.get("response") != "success":
                return RoutingDecision(
                    route=Route.UNKNOWN,
                    superhero_name=decision.superhero_name,
                    reason="The both route was proposed, but the superhero could not be verified."
                )

        except RuntimeError:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The both route was proposed, but the Superhero API could not be reached."
            )

        return RoutingDecision(
            route=Route.BOTH,
            superhero_name=decision.superhero_name,
            reason="Both the text source and Superhero API verified the proposed route."
        )

    # UNKNOWN remains UNKNOWN
    return decision

def verify_route(
    question: str,
    decision: RoutingDecision
) -> RoutingDecision:

    # Verify TEXT source
    if decision.route == Route.TEXT:

        text_results = retrieve_text(question)

        if not text_results:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="The text route was proposed, but no relevant text was found."
            )

        return RoutingDecision(
            route=Route.TEXT,
            superhero_name=None,
            reason="The proposed text route was verified by the text retriever."
        )

    # Verify SUPERHERO source
    if decision.route == Route.SUPERHERO:

        if not decision.superhero_name:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="The superhero route was proposed, but no superhero name was identified."
            )

        try:
            superhero_data = search_superhero(
                decision.superhero_name
            )

            if superhero_data.get("response") != "success":
                return RoutingDecision(
                    route=Route.UNKNOWN,
                    superhero_name=decision.superhero_name,
                    reason="The superhero could not be verified using the Superhero API."
                )

            return RoutingDecision(
                route=Route.SUPERHERO,
                superhero_name=decision.superhero_name,
                reason="The proposed superhero route was verified using the Superhero API."
            )

        except RuntimeError:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The Superhero API could not be reached."
            )

    # Verify BOTH sources
    if decision.route == Route.BOTH:

        if not decision.superhero_name:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="The both route was proposed, but no superhero name was identified."
            )

        # Verify text source
        text_results = retrieve_text(question)

        if not text_results:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The both route was proposed, but no relevant text was found."
            )

        # Verify superhero source
        try:
            superhero_data = search_superhero(
                decision.superhero_name
            )

            if superhero_data.get("response") != "success":
                return RoutingDecision(
                    route=Route.UNKNOWN,
                    superhero_name=decision.superhero_name,
                    reason="The both route was proposed, but the superhero could not be verified."
                )

        except RuntimeError:
            return RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The both route was proposed, but the Superhero API could not be reached."
            )

        return RoutingDecision(
            route=Route.BOTH,
            superhero_name=decision.superhero_name,
            reason="Both the text source and Superhero API verified the proposed route."
        )

    # UNKNOWN remains UNKNOWN
    return decision

def hybrid_route(question: str) -> RoutingDecision:

    # Stage 1: LLM proposes the route
    llm_decision = llm_route(question)

    # Stage 2: deterministic verification
    verified_decision = verify_route(
        question,
        llm_decision
    )

    return verified_decision

def get_superhero_context(superhero_name: str) -> dict:
    """
    Retrieve superhero information from the Superhero API.
    """

    data = search_superhero(superhero_name)

    if data.get("response") != "success":
        raise RuntimeError(
            f"Superhero '{superhero_name}' was not found."
        )

    results = data.get("results", [])

    if not results:
        raise RuntimeError(
            f"No results found for superhero '{superhero_name}'."
        )

    exact_matches = [
        result
        for result in results
        if result.get("name", "").lower() == superhero_name.lower()
    ]

    superhero = (
        exact_matches[0]
        if exact_matches
        else results[0]
    )

    return superhero

def get_text_context(question: str) -> list:
    """
    Retrieve relevant text chunks for the question.
    """

    results = retrieve_text(
        question=question,
        top_k=3,
        threshold=0.1
    )

    return results
ANSWER_SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information provided
in the source context.

Do not invent facts that are not present in the source context.

If the provided sources do not contain enough information to answer
the question, clearly say that the available sources are insufficient.

Every answer MUST include a "Sources" section that identifies where
the information came from.

For text sources, use the document filename.

For superhero information, identify the source as:
Superhero API.
"""
def generate_answer(
    question: str,
    source_context: str
) -> str:
    """
    Generate the final answer using the hosted LLM.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": ANSWER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

Source context:
{source_context}
"""
            }
        ]
    )

    return response.choices[0].message.content
def answer_question(question: str) -> dict:
    """
    Process a user question through the complete chatbot pipeline.

    Returns the generated answer and the sources used.
    """

    decision = hybrid_route(question)

    if decision.route == Route.UNKNOWN:
        return {
            "answer": (
                "I could not determine which available source "
                "can reliably answer this question."
            ),
            "sources": []
        }

    source_parts = []
    sources = []

    if decision.route in {Route.TEXT, Route.BOTH}:

        text_results = get_text_context(question)

        for result in text_results:

            source_parts.append(
                f"Source: {result['document']}\n"
                f"Content: {result['content']}"
            )

            if result["document"] not in sources:
                sources.append(result["document"])

    if decision.route in {Route.SUPERHERO, Route.BOTH}:

        superhero = get_superhero_context(
            decision.superhero_name
        )

        source_parts.append(
            "Source: Superhero API\n"
            f"Name: {superhero.get('name')}\n"
            f"Powerstats: {superhero.get('powerstats')}\n"
            f"Biography: {superhero.get('biography')}\n"
            f"Appearance: {superhero.get('appearance')}\n"
            f"Work: {superhero.get('work')}\n"
            f"Connections: {superhero.get('connections')}"
        )

        sources.append("Superhero API")

    source_context = "\n\n".join(source_parts)

    answer = generate_answer(
        question=question,
        source_context=source_context
    )

    return {
        "answer": answer,
        "sources": sources
    }
    
class AskRequest(BaseModel):
    question: str = Field(
        description="Natural language question for the chatbot"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
app = FastAPI(
    title="AI Engineer Assessment Chatbot",
    description="Hybrid chatbot using a text dataset, Superhero API, and hosted LLM.",
    version="1.0.0"
)

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question is too long. Maximum length is 1000 characters."
        )

    try:
        result = answer_question(question)

        return AskResponse(
            question=question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the question."
        )
        
