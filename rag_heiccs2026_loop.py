import time
import os
import glob
import re
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

"""
This script implements a RAG (Retrieval-Augmented Generation) workflow
to automatically generate household financial advice.

1. Load environment variables (HF_TOKEN, PINECONE_API_KEY, OPENAI_API_KEY) from `token.env`.
2. Vectorize documents using LangChain's OpenAIEmbeddings (text-embedding-3-small),
   and initialize Pinecone index "heiccs2026" as the vector store.
3. Load the Japanese model Llama-3-ELYZA-JP-8B from Hugging Face Transformers
   and set up a `pipeline("text-generation")` (e.g., configure max_length, temperature as needed).
4. Wrap the pipeline with LangChain's `HuggingFacePipeline`, combine it with a retriever
   (top_k=1) on the vector store, and build a RetrievalQA chain.
5. Read household-consultation prompts, run the chain to perform retrieve + generate.
6. Output generated advice and processing time to the console.
"""

# ── Start total timer ──
total_start = time.time()

# Start per-run timer (optional)
start_time = time.time()

# Load environment variables from .env
load_dotenv("token.env")
HF_TOKEN = os.getenv("HF_TOKEN")

model_name = "elyza/Llama-3-ELYZA-JP-8B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Load model (quantization disabled; device_map="auto" will use available device)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", token=HF_TOKEN
)

# Initialize embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_name = "heiccs2026"

# Initialize Pinecone vector store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Build text-generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define LLM via HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=text_generator)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Configure RetrievalQA chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ── Read prompts from external files (prompt.txt) ──
# (This block has been replaced by the loop below.)

# ── Batch-process prompts from a folder (`edit_prompt`) ──
BASE_DIR = os.path.dirname(__file__)
PROMPT_DIR = os.path.join(BASE_DIR, "edit_prompt")
ADV_DIR = os.path.join(BASE_DIR, "advise")

# Collect prompt files like prompt1-4.txt, prompt2-3.txt, ... and sort by embedded numbers
prompt_files = sorted(
    glob.glob(os.path.join(PROMPT_DIR, "prompt*-p*.txt")),
    key=lambda p: tuple(map(int, re.findall(r"\d+", os.path.basename(p)))),
)[:5]  # Process only the first 5 files (change or remove this slice as needed)

for pf in prompt_files:
    # Build output filename
    base = os.path.splitext(os.path.basename(pf))[0]  # e.g., "prompt1-4"
    out_name = base.replace("prompt", "advise") + ".txt"  # e.g., "advise1-4.txt"
    out_path = os.path.join(ADV_DIR, out_name)

    # Skip if output already exists
    if os.path.exists(out_path):
        print(f"Skip {out_name} (already exists)")
        continue

    # Read prompt
    with open(pf, encoding="utf-8") as f:
        prompt_text = f.read()

    # Invoke the chain
    start = time.time()
    response = chain.invoke(prompt_text)  # pass raw string directly
    result_txt = response["result"]  # extract response text

    # Write result to file
    os.makedirs(ADV_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_txt)

    # Show elapsed time for this file
    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)
    print(f"{out_name}: {m} min {s} sec (saved)")

# ── After loop: show total elapsed time ──
total_elapsed = time.time() - total_start
tm, ts = divmod(int(total_elapsed), 60)
print("=== All files processed ===")
print(f"Total elapsed: {tm} min {ts} sec")
