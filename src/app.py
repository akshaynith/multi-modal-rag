import os, json, uuid, base64
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# LangChain & Chroma (modern imports)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage

# ---------- helpers ----------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_retriever(collection: str = "mmrag_ui_collection") -> MultiVectorRetriever:
    vs = Chroma(collection_name=collection, embedding_function=OpenAIEmbeddings())
    parent_store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vs, docstore=parent_store, id_key="doc_id")
    return retriever

def index_all(
    retriever: MultiVectorRetriever,
    ingested: Dict[str, Any],
    summaries: Dict[str, Any],
):
    """
    Child vectors:
      - text summary per page  -> metadata: kind='text', page
      - image caption per image-> metadata: kind='image', image_index
    Parents:
      - text parent: {'type':'text','page':int,'text':str}
      - image parent:{'type':'image','index':int,'b64':str}
    """
    texts = ingested.get("texts", [])
    images_b64 = ingested.get("images_b64", [])
    text_summaries = summaries.get("text_summaries", [])
    image_summaries = summaries.get("image_summaries", [])

    page_to_text = {t["page"]: t["text"] for t in texts if "page" in t and "text" in t}

    # text children
    for ts in text_summaries:
        page = ts["page"]
        summary = ts["summary"]
        parent_payload = page_to_text.get(page, "")
        parent_id = str(uuid.uuid4())
        retriever.vectorstore.add_documents([
            Document(page_content=summary, metadata={"doc_id": parent_id, "kind": "text", "page": page})
        ])
        retriever.docstore.mset([(parent_id, {"type": "text", "page": page, "text": parent_payload})])

    # image children
    for i, cap in enumerate(image_summaries):
        parent_id = str(uuid.uuid4())
        retriever.vectorstore.add_documents([
            Document(page_content=cap, metadata={"doc_id": parent_id, "kind": "image", "image_index": i})
        ])
        if i < len(images_b64):
            retriever.docstore.mset([(parent_id, {"type": "image", "index": i, "b64": images_b64[i]})])

def separate_context(parents: List[Any]) -> Dict[str, List[Any]]:
    out = {"texts": [], "images_b64": [], "meta": []}
    for p in parents:
        # p is whatever we stored via docstore
        if isinstance(p, dict) and p.get("type") == "text":
            out["texts"].append(p.get("text", ""))
            out["meta"].append({"type": "text", "page": p.get("page")})
        elif isinstance(p, dict) and p.get("type") == "image":
            b64 = p.get("b64")
            if b64:
                out["images_b64"].append(b64)
                out["meta"].append({"type": "image", "index": p.get("index")})
    return out

def build_multimodal_message(question: str, ctx: Dict[str, List[Any]], max_images: int = 2):
    # join text snippets (trim to keep tokens reasonable)
    text_snips = []
    for t in ctx["texts"][:4]:
        if isinstance(t, str) and t.strip():
            text_snips.append(t[:1500])
    context_text = "\n---\n".join(text_snips)

    content = [
        {"type": "text", "text":
            "Answer strictly using the provided context.\n\n"
            "If any images are attached, add a short section titled 'Figure insights' "
            "explaining what the figure shows (axes, units, trends) and how it supports the answer.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        }
    ]
    for b64 in ctx["images_b64"][:max_images]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [HumanMessage(content=content)]

# ---------- UI ----------
def main():
    st.set_page_config(page_title="Multimodal RAG Viewer", layout="wide")
    st.title("📄🔎 Multimodal RAG — Viewer")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY missing. Add it to your .env and restart.")
        st.stop()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.subheader("Artifacts")
        ingested_path = st.text_input("Ingested JSON", "./content/ingested.json")
        summaries_path = st.text_input("Summaries JSON", "./content/summaries.json")
        collection = st.text_input("Chroma collection", "mmrag_ui_collection")
        k = st.slider("Top-K (child vectors)", min_value=2, max_value=12, value=6, step=1)
        max_imgs = st.slider("Max images in prompt", min_value=0, max_value=4, value=2, step=1)

        if st.button("Build (or rebuild) index"):
            try:
                ing = read_json(ingested_path)
                summ = read_json(summaries_path)
                st.session_state.retriever = build_retriever(collection)
                # clear vector store by recreating (simplest for demo)
                index_all(st.session_state.retriever, ing, summ)
                st.success("Index built ✅")
            except Exception as e:
                st.exception(e)

        st.subheader("Ask a question")
        q = st.text_area("Question", "What is multi-head attention?")
        go = st.button("Run query")

    with col_r:
        st.subheader("Answer")
        if go:
            if "retriever" not in st.session_state:
                st.warning("Build the index first.")
            else:
                try:
                    # fetch parents (MultiVectorRetriever.invoke uses retriever.search_kwargs if set)
                    # Set search_kwargs here dynamically:
                    st.session_state.retriever.search_kwargs = {"k": k}
                    parents = st.session_state.retriever.invoke(q)
                    ctx = separate_context(parents)

                    # show retrieved context (text snippets)
                    with st.expander("Retrieved text context", expanded=False):
                        for i, t in enumerate(ctx["texts"][:4], start=1):
                            st.markdown(f"**Snippet {i}**")
                            st.write(t[:1200])

                    # show retrieved images
                    if ctx["images_b64"]:
                        st.markdown("**Retrieved figures**")
                        img_cols = st.columns(min(3, len(ctx["images_b64"])))
                        for i, b64 in enumerate(ctx["images_b64"][:6]):
                            img_bytes = base64.b64decode(b64)
                            img_cols[i % len(img_cols)].image(img_bytes, caption=f"retrieved_image_{i+1}.jpg", use_container_width=True)
                    else:
                        st.info("No figures retrieved for this query.")

                    # build prompt and answer
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                    msg = build_multimodal_message(q, ctx, max_images=max_imgs)
                    resp = llm.invoke(msg)

                    st.markdown("### Final Answer")
                    st.write(resp.content)
                except Exception as e:
                    st.exception(e)

    st.caption("Tip: tweak Top-K and Max Images and re-run. Use precise questions to pull different context.")

if __name__ == "__main__":
    main()
