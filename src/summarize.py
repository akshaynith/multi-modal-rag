import os, json, argparse, math
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---------- helpers ----------
def load_ingested(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def batch(iterable, size: int):
    for i in range(0, len(iterable)):
        if i % size == 0:
            yield iterable[i:i+size]

# ---------- build chains ----------
def build_text_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    prompt = ChatPromptTemplate.from_template(
        "You are an expert technical writer. Summarize the following content "
        "concisely and information-dense, preserving key facts, entities, and quantitative details. "
        "Avoid fluff. Output only the summary.\n\nCONTENT:\n{element}"
    )
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return {"element": lambda x: x} | prompt | llm | StrOutputParser()

def build_image_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    # Multimodal: text + inline base64 image
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text":
             "Describe the image like a caption for retrieval: spell out axes/titles/legends, units, key trends, and what the figure supports. Output only the caption."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}" }}
        ])
    ])
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return prompt | llm | StrOutputParser()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Lesson 2: Summarize texts and images with OpenAI")
    ap.add_argument("--injson", default="./content/ingested.json", help="Path to ingested.json from Lesson 1")
    ap.add_argument("--outjson", default="./content/summaries.json", help="Where to write summaries JSON")
    ap.add_argument("--max_text_chars", type=int, default=6000, help="Truncate long page text to this many chars")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for API calls")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Put it in .env")

    data = load_ingested(args.injson)
    texts = data.get("texts", [])          # [{"page": int, "text": str}, ...]
    images_b64 = data.get("images_b64", [])# [str, ...]

    # Build chains
    text_chain = build_text_chain(args.model)
    img_chain  = build_image_chain(args.model)

    # Prepare inputs (truncate to control cost)
    text_inputs: List[str] = []
    text_meta:   List[Dict[str, Any]] = []
    for t in texts:
        page = t.get("page")
        raw  = (t.get("text") or "")
        text_inputs.append(raw[:args.max_text_chars])
        text_meta.append({"page": page, "orig_len": len(raw)})

    # Summarize texts in batches
    text_summaries: List[Dict[str, Union[int, str]]] = []
    if text_inputs:
        for chunk in batch(text_inputs, args.batch):
            text_summaries.extend(text_chain.batch(chunk, {"max_concurrency": min(len(chunk), args.batch)}))
    # attach meta
    text_summaries = [
        {"page": text_meta[i]["page"], "summary": s, "orig_len": text_meta[i]["orig_len"]}
        for i, s in enumerate(text_summaries)
    ]

    # Summarize images (if any)
    image_summaries: List[str] = []
    if images_b64:
        for chunk in batch(images_b64, args.batch):
            # the prompt expects dicts like {"image_b64": <b64>}
            image_inputs = [{"image_b64": b64} for b64 in chunk]
            image_summaries.extend(img_chain.batch(image_inputs, {"max_concurrency": min(len(chunk), args.batch)}))

    # Write out
    out = {
        "text_summaries": text_summaries,     # [{page, summary, orig_len}]
        "image_summaries": image_summaries,   # [str]
        "counts": {"texts": len(texts), "images": len(images_b64)}
    }
    os.makedirs(os.path.dirname(args.outjson), exist_ok=True)
    with open(args.outjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Summarized {len(text_summaries)} text chunks and {len(image_summaries)} images.")
    if text_summaries:
        print("\n--- Sample text summary (page {p}) ---\n{t}\n".format(
            p=text_summaries[0]['page'], t=text_summaries[0]['summary'][:600]))
    if image_summaries:
        print("\n--- Sample image summary ---\n" + image_summaries[0][:400])

if __name__ == "__main__":
    main()
