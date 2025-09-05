# fa-rag-hie-repro

This repository contains a minimal RAG (Retrieval-Augmented Generation) pipeline
for generating household financial advice in Japanese.

## How to Run

1. Set API keys in `token.env`:
   HF_TOKEN=xxxx
   OPENAI_API_KEY=xxxx
   PINECONE_API_KEY=xxxx

2. Prepare data:
   - Create a Pinecone index named `heiccs2026` and insert documents with
     embeddings (text-embedding-3-small).
   - Place prompt files under `edit_prompt/` (e.g., prompt1-1.txt).
   - Generated results will be saved in `advise/`.

3. Run the script:
   python rag_heiccs2026_loop.py

## Notes

- Do not commit `token.env` (keep your API keys safe).
- Output directory `advise/` will be created automatically if missing.
- Example prompt files are under `prompt/sample/`.
  Put your own prompt files under `edit_prompt/` when running the script.

## License
Apache License 2.0
