import os, re, json
import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import fuzz
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ===== 0. HÀM HỖ TRỢ =====
def remove_vietnamese_diacritics(text: str) -> str:
    return unidecode(text).lower()

def pretty(text: str) -> str:
    text = re.sub(r"([;:.\)\]\}])\s*(?=[\+\-•–])", r"\1\n", text)
    text = re.sub(r"\s*([+\-•–])\s+", r"\n\1 ", text)
    text = re.sub(r"\s*(Điều\s+\d+\.)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ===== 1. ĐƯỜNG DẪN & THAM SỐ =====
DB_DIR = "./vector_store"
COLLECTION = "so_tay_hcmue"
CHUNKS_TXT = "./chunks.txt"
TABLE_JSON = "./so_tay_all_tables_clean.json"
COURSE_JSON = "./mon_hoc_mo_ta.json"
MIN_EXPECTED_COUNT = 200
TOP_K = 5

# ===== 2. TẢI DỮ LIỆU JSON =====
with open(TABLE_JSON, "r", encoding="utf-8") as f:
    tables = json.load(f)

with open(COURSE_JSON, "r", encoding="utf-8") as f:
    course_list = json.load(f)

COURSE_DATA = {
    remove_vietnamese_diacritics(item['ten_mon']): {
        "ten_mon": item['ten_mon'],
        "Description": item['Description']
    }
    for item in course_list
}

# ===== 3. KHỞI TẠO MODEL & CHROMA =====
print("🚀 Đang tải model...")
model = SentenceTransformer("keepitreal/vietnamese-sbert")

print("📚 Đang khởi tạo ChromaDB...")
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings())
col = None

def ensure_collection_ready():
    """Đảm bảo collection tồn tại hoặc tạo mới từ chunks.txt"""
    global col
    try:
        col = client.get_collection(COLLECTION)
        total = col.count()
        if total >= MIN_EXPECTED_COUNT:
            return
    except Exception:
        pass

    os.makedirs(DB_DIR, exist_ok=True)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    with open(CHUNKS_TXT, "r", encoding="utf-8") as f:
        docs = [x.strip() for x in f.read().split("\n\n") if len(x.strip()) > 80][:400]  # Giới hạn nhẹ RAM

    metadatas = [{"source": "So_Tay_Chinh"} for _ in docs]
    embs = []
    for i in range(0, len(docs), 32):
        batch = docs[i:i+32]
        batch_emb = model.encode(batch, normalize_embeddings=True).tolist()
        embs.extend(batch_emb)

    col.add(
        ids=[str(i) for i in range(len(docs))],
        documents=docs,
        embeddings=embs,
        metadatas=metadatas
    )

ensure_collection_ready()

# ===== 4. HÀM TRA CỨU =====
def find_table_by_keyword(query: str) -> str | None:
    normalized_query = remove_vietnamese_diacritics(query)
    mapping = {
        "4 sang chu": ["thang_diem_4"],
        "hoc bong": ["xep_loai_hoc_bong"],
        "chu sang 10": ["thang_diem_10_chu"],
        "xep loai hoc luc": ["xep_loai_hoc_luc"],
        "yeu cau hoc bong": ["yeu_cau_hoc_bong"],
        "diem ren luyen": ["diem_ren_luyen1", "diem_ren_luyen2", "diem_ren_luyen3"]
    }

    BEST_MATCH_SCORE = 90
    best_key, best_score = None, 0
    for k in mapping.keys():
        nk = remove_vietnamese_diacritics(k)
        score = max(
            fuzz.WRatio(normalized_query, nk),
            fuzz.partial_ratio(normalized_query, nk),
            fuzz.token_set_ratio(normalized_query, nk)
        )
        if score > best_score and score >= BEST_MATCH_SCORE:
            best_key, best_score = k, score

    if not best_key:
        return None

    out = [f"### 📊 Bảng tra cứu (Độ khớp: {best_score}%)"]
    for type_name in mapping[best_key]:
        for t in tables:
            if t.get("type") == type_name:
                df = pd.DataFrame(t["data"])
                title = t.get("title", type_name)
                out.append(f"\n#### {title}\n" + df.to_markdown(index=False))
    return "\n".join(out) if len(out) > 1 else None

def find_course_by_fuzzy_match(query: str):
    qn = remove_vietnamese_diacritics(query)
    best_score, best = 0, None
    for k, c in COURSE_DATA.items():
        score = max(
            fuzz.token_set_ratio(qn, k),
            fuzz.WRatio(qn, k),
            fuzz.partial_ratio(qn, k)
        )
        if score > best_score:
            best_score, best = score, c
    if best and best_score >= 90:
        return f"📚 **Môn học:** {best['ten_mon']}\n\n{best['Description']}\n_(Độ khớp: {best_score}%)_"
    return None

def query_vector_db(query: str):
    q_emb = model.encode(query, normalize_embeddings=True).tolist()
    res = col.query(
        query_embeddings=[q_emb],
        include=["documents", "metadatas"],
        where={"source": {"$eq": "So_Tay_Chinh"}},
        n_results=TOP_K
    )
    docs = (res.get("documents") or [[]])[0]
    return [pretty(d) for d in docs if d.strip()]

# ===== 5. KHỞI TẠO FASTAPI =====
app = FastAPI(title="🎓 Chatbot Tra cứu HCMUE", description="API tìm kiếm nội dung Sổ tay & Môn học", version="1.0")

@app.get("/")
def root():
    return {"message": "✅ Chatbot HCMUE API đang hoạt động!"}

@app.post("/query")
async def query_api(request: Request):
    data = await request.json()
    query = data.get("question", "").strip()
    if not query:
        return {"error": "⚠️ Thiếu trường 'question'."}

    # 1) Tra bảng
    tb = find_table_by_keyword(query)
    if tb:
        return {"type": "table", "result": tb}

    # 2) Tra môn học
    course_rs = find_course_by_fuzzy_match(query)
    if course_rs:
        return {"type": "course", "result": course_rs}

    # 3) Vector search
    docs = query_vector_db(query)
    if not docs:
        return {"type": "none", "result": "❌ Không tìm thấy thông tin phù hợp."}

    return {"type": "vector", "result": docs}

# ===== 6. CHẠY APP =====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
