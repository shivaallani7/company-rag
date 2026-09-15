"""
Shared embedding model for ingest.py and query.py.

Uses HuggingFaceInstructEmbeddings (InstructorEmbedding + sentence-transformers
under the hood), so everything runs fully locally -- no API key, no data
leaving your machine.

The FIRST run downloads the model (~1.5GB for instructor-large) from
Hugging Face and caches it under ~/.cache/torch (or ~/.cache/huggingface).
Every run after that is fast and fully offline.

If instructor-large is too slow/heavy for your machine, swap MODEL_NAME
for "hkunlp/instructor-base" (much smaller, still solid quality).
"""
import os
import huggingface_hub

# sentence-transformers==2.2.2 (pinned for InstructorEmbedding compatibility,
# see requirements.txt) still imports the long-removed `cached_download` from
# huggingface_hub, and calls it with its old (url, cache_dir, force_filename, ...)
# signature. Newer huggingface_hub versions (required by transformers) dropped
# that function in favor of `hf_hub_download(repo_id, filename, ...)`, so
# restore a compatible shim here rather than fighting a three-way pin conflict
# between sentence-transformers / huggingface_hub / transformers.
if not hasattr(huggingface_hub, "cached_download"):
    def _cached_download(url, cache_dir=None, force_filename=None, use_auth_token=None, **_ignored):
        import requests

        os.makedirs(cache_dir, exist_ok=True)
        dest = os.path.join(cache_dir, force_filename or os.path.basename(url))
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        headers = {}
        token = use_auth_token if isinstance(use_auth_token, str) else None
        if token:
            headers["authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest

    huggingface_hub.cached_download = _cached_download

from langchain_community.embeddings import HuggingFaceInstructEmbeddings

MODEL_NAME = "hkunlp/instructor-large"

_embeddings = None


def get_embeddings():
    """Return a process-wide cached HuggingFaceInstructEmbeddings instance.

    ingest.py and query.py both call this so the exact same model/instructions
    are used to embed documents and queries -- if these ever diverge,
    similarity search quality silently degrades.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceInstructEmbeddings(
            model_name=MODEL_NAME,
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )
    return _embeddings
