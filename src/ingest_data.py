import fitz  # PyMuPDF
import base64, argparse, os, json

def extract(pdf_path):
    doc = fitz.open(pdf_path)
    texts, images_b64 = [], []
    for page_index in range(len(doc)):
        page = doc[page_index]
        # embedded text
        t = page.get_text("text")
        if t and t.strip():
            texts.append({"page": page_index+1, "text": t})
        # images
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("jpeg")
            images_b64.append(base64.b64encode(img_bytes).decode("utf-8"))
            pix = None
    return texts, images_b64

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--outjson", default="./content/ingested.json")
    ap.add_argument("--dump-examples", action="store_true")
    ap.add_argument("--outdir", default="./content")
    args = ap.parse_args()

    texts, images = extract(args.pdf)
    print(f"Texts: {len(texts)}")
    print(f"Images: {len(images)}")

    # save JSON for next lesson
    os.makedirs(os.path.dirname(args.outjson), exist_ok=True)
    with open(args.outjson, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "images_b64": images}, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON -> {args.outjson}")

    if texts:
        print("\n--- First text snippet ---")
        print(texts[0]["text"][:600])

    if args.dump_examples and images:
        os.makedirs(args.outdir, exist_ok=True)
        for i, b64 in enumerate(images[:2], start=1):
            with open(os.path.join(args.outdir, f"fallback_image_{i}.jpg"), "wb") as f:
                f.write(base64.b64decode(b64))
        print(f"\nSaved sample images to {args.outdir}")
