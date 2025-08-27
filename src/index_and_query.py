import os, json, argparse, base64, uuid
from typing import Any, Dict, List

from dotenv import load_dotenv

# LangChain - vector & retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage

# ---------- IO ----------
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Build retriever ----------
def build_retriever(collection: str = "mmrag_collection") -> MultiVectorRetriever:
    vs = Chroma(collection_name=collection, embedding_function=OpenAIEmbeddings())
    parent_store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vs, docstore=parent_store, id_key="doc_id")
    return retriever

# ---------- Index ----------
def index_all(
    retriever: MultiVectorRetriever,
    ingested: Dict[str, Any],
    summaries: Dict[str, Any],
):
    """
    Child vectors:
      - text summary per page
      - image captions (order matches ingested images)
    Parents:
      - page full text (for text child -> parent)
      - base64 image payload (for image child -> parent)
    """
    texts = ingested.get("texts", [])              # [{page:int, text:str}, ...]
    images_b64 = ingested.get("images_b64", [])    # [str, ...]
    text_summaries = summaries.get("text_summaries", [])  # [{page:int, summary:str, orig_len:int}, ...]
    image_summaries = summaries.get("image_summaries", [])# [str, ...]

    # Map page->full text for quick lookup
    page_to_text = {t["page"]: t["text"] for t in texts if "page" in t and "text" in t}

    # 1) Upsert text summaries
    for ts in text_summaries:
        page = ts["page"]
        summary = ts["summary"]
        parent_payload = page_to_text.get(page, "")  # full page text
        parent_id = str(uuid.uuid4())
        # child vector
        retriever.vectorstore.add_documents([
            Document(page_content=summary, metadata={"doc_id": parent_id, "kind": "text", "page": page})
        ])
        # parent store
        retriever.docstore.mset([(parent_id, {"type": "text", "page": page, "text": parent_payload})])

    # 2) Upsert image captions
    # We assume order alignment: image_summaries[i] describes images_b64[i]
    for i, cap in enumerate(image_summaries):
        parent_id = str(uuid.uuid4())
        # child vector
        retriever.vectorstore.add_documents([
            Document(page_content=cap, metadata={"doc_id": parent_id, "kind": "image", "image_index": i})
        ])
        # parent store (raw base64 is the parent payload)
        if i < len(images_b64):
            retriever.docstore.mset([(parent_id, {"type": "image", "index": i, "b64": images_b64[i]})])

# ---------- Query ----------
def separate_context(parents: List[Any]) -> Dict[str, List[Any]]:
    out = {"texts": [], "images_b64": []}
    for p in parents:
        # We stored dicts as parents
        if isinstance(p, dict) and p.get("type") == "text":
            out["texts"].append(p.get("text", ""))
        elif isinstance(p, dict) and p.get("type") == "image":
            b64 = p.get("b64")
            if b64:
                out["images_b64"].append(b64)
    return out

def build_multimodal_message(question: str, ctx: Dict[str, List[Any]], max_images: int = 2):
    # Concatenate a few text snippets
    text_snippets = []
    for t in ctx["texts"][:4]:
        if isinstance(t, str):
            text_snippets.append(t[:1500])
    context_text = "\n---\n".join(text_snippets)

    content = [
        {"type": "text", "text": f"Answer strictly using the provided context.\n\nContext:\n{context_text}\n\nQuestion: {question}"}
    ]
    for b64 in ctx["images_b64"][:max_images]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [HumanMessage(content=content)]

def answer_question(retriever: MultiVectorRetriever, question: str) -> str:
    parents = retriever.get_relevant_documents(question)  # returns parents (InMemoryStore payloads)
    ctx = separate_context(parents)

    # --- Save retrieved images for inspection ---
    import os, base64
    out_dir = "./content"
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, b64 in enumerate(ctx["images_b64"][:2], start=1):
        path = os.path.join(out_dir, f"retrieved_image_{i}.jpg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        saved.append(path)
    if saved:
        print("[images] saved to:")
        for p in saved:
            print("  -", p)
    # --------------------------------------------

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    msg = build_multimodal_message(question, ctx, max_images=2)
    resp = llm.invoke(msg)

    # Print a tiny provenance preview
    print("\n=== Retrieved preview ===")
    for i, t in enumerate(ctx["texts"][:2], start=1):
        preview = t[:300].replace("\n", " ")
        print(f"[text {i}] {preview}\n")
    if ctx["images_b64"]:
        print(f"[images] attached: {min(len(ctx['images_b64']), 2)}")
    return resp.content


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Lesson 3: Index & Query")
    ap.add_argument("--ingested", default="./content/ingested.json")
    ap.add_argument("--summaries", default="./content/summaries.json")
    ap.add_argument("--question", default="What is multi-head attention?")
    ap.add_argument("--collection", default="mmrag_collection")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    ingested = read_json(args.ingested)
    summaries = read_json(args.summaries)

    retriever = build_retriever(args.collection)
    index_all(retriever, ingested, summaries)

    print("Indexed. Running a sample query…")
    ans = answer_question(retriever, args.question)
    print("\n=== Answer ===\n" + ans)

if __name__ == "__main__":
    main()
