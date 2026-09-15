"""
Ingest every PDF in pdfs/ into a local, persistent Chroma vector store.

Usage:
    python ingest.py

Re-run this any time you add/remove/change PDFs in pdfs/ -- it rebuilds
the index from scratch (simple and safe; fine for a "few dozen PDFs" scale
RAG like this one).
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from embeddings import get_embeddings

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "company_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents(pdf_dir: Path):
    """Load every *.pdf in pdf_dir into a flat list of LangChain Documents.

    Each page becomes its own Document. metadata["source"] is set to the
    PDF's filename (PyPDFLoader already sets metadata["page"]) so answers
    can be traced back to "which PDF, which page".
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}.\n"
            "Drop your Company Onboarding / Social Engineering PDFs into "
            "that folder and re-run `python ingest.py`."
        )

    documents = []
    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)
        print(f"  -> {len(pages)} pages")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks, embeddings, persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"Embedding {len(chunks)} chunks and writing to {persist_dir} "
          "(this can take a while on first run) ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    return vectorstore


def main():
    documents = load_documents(PDF_DIR)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    build_vectorstore(chunks, embeddings, PERSIST_DIR)
    print(f"\nDone. Indexed {len(chunks)} chunks from {len(list(PDF_DIR.glob('*.pdf')))} "
          f"PDF(s) into {PERSIST_DIR}")
    print("Now run: python query.py \"your question here\"")


if __name__ == "__main__":
    main()
