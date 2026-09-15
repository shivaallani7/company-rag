# company-rag

A basic local RAG (Retrieval-Augmented Generation) pipeline over your
Company Onboarding and Social Engineering PDFs. Everything runs on your
machine -- no API keys, nothing uploaded anywhere.

Stack:
- `langchain_community.document_loaders.PyPDFLoader` -- reads PDFs page by page
- `langchain.text_splitter.RecursiveCharacterTextSplitter` -- chunks the text
- `langchain_community.embeddings.HuggingFaceInstructEmbeddings` -- local embeddings
- `langchain_community.vectorstores.Chroma` -- local, persistent vector store

This currently builds and tests the **retrieval** half only (load -> chunk
-> embed -> store -> search). The **generation** half (an LLM turning
retrieved passages into a written answer) is stubbed out in
`generate_answer()` in `query.py` -- see that file for how to wire up
Ollama, OpenAI, or Anthropic when you're ready.

## Setup

```bash
cd company-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note on `sentence-transformers`: it's pinned to `2.2.2` because
`InstructorEmbedding` (the package behind `HuggingFaceInstructEmbeddings`)
relies on an internal API that newer `sentence-transformers` releases
removed. If you hit an import error mentioning `SentenceTransformer`, this
pin is why -- don't upgrade it.

## 1. Add your PDFs

Copy your Company Onboarding and Social Engineering PDFs into `pdfs/`.

## 2. Build the index

```bash
python ingest.py
```

First run downloads the `instructor-large` embedding model (~1.5GB) from
Hugging Face and caches it locally -- after that, it's fully offline. This
writes a persistent Chroma DB into `chroma_db/`. Re-run this any time you
add, remove, or change PDFs in `pdfs/` (it rebuilds the index from scratch).

If `instructor-large` is too slow/heavy for your machine, open
`embeddings.py` and change `MODEL_NAME` to `"hkunlp/instructor-base"`.

## 3. Ask questions (retrieval only, for now)

```bash
python query.py "What's covered in the first week of onboarding?"
python query.py "What are common social engineering red flags?"
```

Or run `python query.py` with no arguments for an interactive prompt.

Each result shows the top-matching passages with their source PDF and page
number, plus a placeholder line noting no LLM is configured yet.

## 4. (Next step) Wire up generation

Open `query.py` and fill in `generate_answer()`. It already has commented
examples for:
- **Ollama** (fully local, e.g. `ollama pull llama3` then
  `langchain_community.llms.Ollama(model="llama3")`)
- **OpenAI** (`langchain_openai.ChatOpenAI`)
- **Anthropic** (`langchain_anthropic.ChatAnthropic`)

## Files

- `pdfs/` -- put your source PDFs here
- `ingest.py` -- builds the Chroma index from `pdfs/`
- `query.py` -- retrieves relevant chunks for a question (generation stubbed)
- `embeddings.py` -- shared embedding model used by both scripts
- `chroma_db/` -- generated on first `ingest.py` run (persistent vector store)
- `requirements.txt` -- pinned dependencies
