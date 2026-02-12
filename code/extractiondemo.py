import os
import re
import fitz
import pdfplumber

OUTPUT_DIR = "img"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# distance helper
# ----------------------------
def rect_distance(a, b):
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return (dx*dx + dy*dy) ** 0.5


# ----------------------------
# Approach 1: vector → PNG crop
# ----------------------------
def extract_vector_diagrams(page, page_num):

    drawings = page.get_drawings()
    if not drawings:
        return 0

    boxes = [d["rect"] for d in drawings]
    merged = []

    for r in boxes:
        added = False
        for i, m in enumerate(merged):
            if r.intersects(m) or rect_distance(r, m) < 30:
                merged[i] = m | r
                added = True
                break
        if not added:
            merged.append(r)

    count = 0

    for rect in merged:

        if rect.width * rect.height < 15000:
            continue

        pix = page.get_pixmap(clip=rect, dpi=300)

        fname = f"{OUTPUT_DIR}/page_{page_num:03}_diag_{count+1}.png"
        pix.save(fname)

        count += 1

    return count


# ----------------------------
# Approach 3: caption crop
# ----------------------------
def extract_caption_diagrams(plumber_page, page_num):

    pattern = re.compile(r"Fig\.?\s*\d+(\.\d+)*", re.I)

    words = plumber_page.extract_words()
    im = plumber_page.to_image(resolution=300)

    count = 0

    for w in words:
        if pattern.search(w["text"]):

            top = w["top"]

            crop = im.crop((
                0,
                max(0, top - 350),
                plumber_page.width,
                top - 20
            ))

            fname = f"{OUTPUT_DIR}/page_{page_num:03}_caption_{count+1}.png"
            crop.save(fname)

            count += 1

    return count


# ----------------------------
# main
# ----------------------------
def extract_diagrams_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    plumber = pdfplumber.open(pdf_path)

    total = 0

    for i in range(len(doc)):
        print(f"Processing page {i+1}")

        v = extract_vector_diagrams(doc[i], i)

        if v > 0:
            total += v
        else:
            total += extract_caption_diagrams(plumber.pages[i], i)

    plumber.close()

    print(f"\nDone. Extracted {total} diagrams → img/")


if __name__ == "__main__":
    extract_diagrams_from_pdf("./pdf/keph201.pdf")