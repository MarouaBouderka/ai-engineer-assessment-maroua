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
            timeout=30
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
def verify_route(question: str, decision: RoutingDecision):
    """
    Verify the LLM routing decision using the actual data sources.

    Returns:
        verified_decision: The verified routing decision.
        retrieved_text: Text retrieval results, if available.
        superhero_data: Superhero API result, if available.
    """

    retrieved_text = []
    superhero_data = None

    if decision.route == Route.TEXT:
        retrieved_text = retrieve_text(
            question,
            top_k=3,
            threshold=0.1
        )

        if retrieved_text:
            return decision, retrieved_text, superhero_data

        return (
            RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=None,
                reason="No sufficiently relevant text information was found."
            ),
            retrieved_text,
            superhero_data
        )

    if decision.route == Route.SUPERHERO:
        if not decision.superhero_name:
            return (
                RoutingDecision(
                    route=Route.UNKNOWN,
                    superhero_name=None,
                    reason="No superhero name was identified."
                ),
                retrieved_text,
                superhero_data
            )

        try:
            superhero_data = search_superhero(decision.superhero_name)

            if superhero_data.get("response") == "success":
                return decision, retrieved_text, superhero_data

        except RuntimeError:
            pass

        return (
            RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="The Superhero API did not return valid information."
            ),
            retrieved_text,
            superhero_data
        )

    if decision.route == Route.BOTH:
        retrieved_text = retrieve_text(
            question,
            top_k=3,
            threshold=0.1
        )

        if decision.superhero_name:
            try:
                superhero_data = search_superhero(
                    decision.superhero_name
                )

                if superhero_data.get("response") != "success":
                    superhero_data = None

            except RuntimeError:
                superhero_data = None

        text_available = bool(retrieved_text)
        superhero_available = superhero_data is not None

        if text_available and superhero_available:
            return decision, retrieved_text, superhero_data

        if text_available:
            return (
                RoutingDecision(
                    route=Route.TEXT,
                    superhero_name=decision.superhero_name,
                    reason="Text information was available, but superhero information was unavailable."
                ),
                retrieved_text,
                superhero_data
            )

        if superhero_available:
            return (
                RoutingDecision(
                    route=Route.SUPERHERO,
                    superhero_name=decision.superhero_name,
                    reason="Superhero information was available, but no relevant text information was found."
                ),
                retrieved_text,
                superhero_data
            )

        return (
            RoutingDecision(
                route=Route.UNKNOWN,
                superhero_name=decision.superhero_name,
                reason="Neither source provided usable information."
            ),
            retrieved_text,
            superhero_data
        )

    return decision, retrieved_text, superhero_data

def hybrid_route(question: str):
    """
    Route a question using the LLM and verify the decision
    using the actual available sources.
    """
    decision = llm_route(question)

    verified_decision, retrieved_text, superhero_data = verify_route(
        question,
        decision
    )

    return verified_decision, retrieved_text, superhero_data

def get_superhero_context(superhero_data: dict, superhero_name: str):
    """
    Extract the most relevant superhero information
    from already retrieved API data.
    """

    if not superhero_data:
        return None

    results = superhero_data.get("results", [])

    if not results:
        return None

    for superhero in results:
        if superhero.get("name", "").lower() == superhero_name.lower():
            return superhero

    return results[0]

def get_text_context(retrieved_text):
    """
    Use already retrieved text results as the context.
    """
    return retrieved_text
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
    source_context: list
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
def answer_question(question: str):
    """
    Route the question, reuse the verified retrieved data,
    and generate the final answer.
    """

    decision, retrieved_text, superhero_data = hybrid_route(question)

    if decision.route == Route.UNKNOWN:
        return {
            "answer": "I could not find enough relevant information in the available sources to answer this question.",
            "sources": []
        }

    source_context = []
    sources = []

    if decision.route in (Route.TEXT, Route.BOTH):
        text_context = get_text_context(retrieved_text)

        if text_context:
            source_context.append({
                "source": "Text Knowledge Base",
                "content": text_context
            })

            sources.extend([
                item["document"]
                for item in text_context
            ])

    if decision.route in (Route.SUPERHERO, Route.BOTH):
        superhero_context = get_superhero_context(superhero_data, decision.superhero_name)

        if superhero_context:
            source_context.append({
                "source": "Superhero API",
                "content": superhero_context
            })

            sources.append("Superhero API")

    answer = generate_answer(
        question,
        source_context
    )

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources))
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
        
