"""
Ask questions against the local Chroma index built by ingest.py.

Usage:
    python query.py "What should I do in my first week?"
    python query.py                       # interactive mode

This is the "R" (retrieval) half of RAG only, on purpose: it finds and
prints the most relevant passages from your PDFs, with their source file
and page number. Wire up an LLM in generate_answer() below to turn this
into full RAG -- a few options are sketched there (Ollama for fully local,
or an API-based model).
"""
import sys
from pathlib import Path

from langchain_community.vectorstores import Chroma

from embeddings import get_embeddings

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "company_docs"

TOP_K = 4


def load_vectorstore():
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"No index found at {PERSIST_DIR}. Run `python ingest.py` first."
        )
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def retrieve(question: str, vectorstore, k: int = TOP_K):
    """Return [(Document, distance_score), ...], closest first."""
    return vectorstore.similarity_search_with_score(question, k=k)


def generate_answer(question: str, results):
    """
    STUB -- no LLM is wired up yet. This just tells you retrieval-only mode
    is active. Replace the body with a real generation call once you pick
    an LLM, e.g.:

    Fully local (Ollama running `ollama pull llama3` first):
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3")

    OpenAI:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")

    Anthropic:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-5")

    Then, in this function:
        context = "\\n\\n".join(doc.page_content for doc, _ in results)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say you don't know.\\n\\n"
            f"Context:\\n{context}\\n\\nQuestion: {question}"
        )
        return llm.invoke(prompt)
    """
    return (
        "[Retrieval-only mode: no LLM configured yet. "
        "See generate_answer() in query.py to plug one in.]"
    )


def ask(question: str, vectorstore=None):
    vectorstore = vectorstore or load_vectorstore()
    results = retrieve(question, vectorstore)

    print(f"\nQuestion: {question}\n")
    print(generate_answer(question, results))
    print("\nTop matching passages:\n" + "-" * 60)
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n[{i}] {source} (page {page}) -- distance {score:.4f}")
        print(doc.page_content.strip()[:500])
    print("-" * 60)
    return results


def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    vectorstore = load_vectorstore()
    print("Loaded index. Type a question (or 'quit' to exit).")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        ask(question, vectorstore)


if __name__ == "__main__":
    main()
