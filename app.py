import os, re, json
import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import fuzz
import gradio as gr

# ========== Embedding & Vector DB ==========
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util

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
DB_DIR = "./vector_store"              # nơi chứa ChromaDB persistent
COLLECTION = "so_tay_hcmue"
CHUNKS_TXT = "./chunks.txt"
TABLE_JSON = "./so_tay_all_tables_clean.json"
COURSE_JSON = "./mon_hoc_mo_ta.json"
MIN_EXPECTED_COUNT = 200               # ngưỡng sanity check
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
model = SentenceTransformer("keepitreal/vietnamese-sbert")

# chroma client
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings())
col = None

def ensure_collection_ready():
    """Đảm bảo collection tồn tại. Nếu chưa có, build từ chunks.txt"""
    global col
    try:
        col = client.get_collection(COLLECTION)
        # nếu có rồi thì kiểm tra số lượng
        total = col.count()
        if total >= MIN_EXPECTED_COUNT:
            return
    except Exception:
        pass

    # Nếu chưa có hoặc thiếu → build mới
    # Thử lấy zip (nếu user upload vector_store.zip) → giải nén
    if not os.path.exists(DB_DIR) and os.path.exists("./vector_store.zip"):
        import zipfile
        with zipfile.ZipFile("./vector_store.zip", "r") as z:
            z.extractall("./")
    # thử lại get_collection
    try:
        col = client.get_collection(COLLECTION)
        if col.count() >= MIN_EXPECTED_COUNT:
            return
    except Exception:
        pass

    # Không có sẵn: build từ chunks.txt
    # (đảm bảo DB_DIR tồn tại)
    os.makedirs(DB_DIR, exist_ok=True)

    # reset instance & tạo mới
    try:
        chromadb.api.client.SharedSystemClient._instance = None
    except Exception:
        pass

    try:
        # xóa nếu đã tồn tại
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(COLLECTION)

    # đọc chunks
    with open(CHUNKS_TXT, "r", encoding="utf-8") as f:
        docs = [x.strip() for x in f.read().split("\n\n") if len(x.strip()) > 80]

    if not docs:
        raise RuntimeError("Không có dữ liệu trong chunks.txt để build vector store.")

    # Tạo metadata
    metadatas = [{"source": "So_Tay_Chinh"} for _ in docs]

    # Encode & add (batch)
    embs = []
    batch_size = 32
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True).tolist()
        embs.extend(batch_emb)

    col.add(
        ids=[str(i) for i in range(len(docs))],
        documents=docs,
        embeddings=embs,
        metadatas=metadatas
    )

ensure_collection_ready()

# ===== 4. HÀM TRA CỨU BẢNG (mapping nhanh) =====
def find_table_by_keyword(query: str) -> str | None:
    normalized_query = remove_vietnamese_diacritics(query)

    mapping = {
        "4 sang chu": ["thang_diem_4"],
        "hoc bong": ["xep_loai_hoc_bong"],
        "chu sang 10": ["thang_diem_10_chu"],
        "thang diem 10 sang chu": ["thang_diem_10_chu"],
        "thang diem 4 sang chu": ["thang_diem_4"],
        "xep loai hoc luc": ["xep_loai_hoc_luc"],
        "xep loai hoc bong": ["xep_loai_hoc_bong"],
        "yeu cau hoc bong": ["yeu_cau_hoc_bong"],
        "diem ren luyen": ["diem_ren_luyen1", "diem_ren_luyen2", "diem_ren_luyen3", "diem_ren_luyen4", "diem_ren_luyen5"],
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

    # Xuất các bảng tương ứng
    out = [f"### 📊 Kết quả Tra cứu Bảng (Độ khớp: {best_score}%)"]
    for type_name in mapping[best_key]:
        for t in tables:
            if t.get("type") == type_name:
                df = pd.DataFrame(t["data"])
                title = t.get("title", type_name.replace("_", " ").title())
                out.append(f"\n#### {title}\n" + df.to_markdown(index=False))
    return "\n".join(out) if len(out) > 1 else None

# ===== 5. Fuzzy match môn học =====
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
        return f"📚 **Môn học:** {best['ten_mon']}\n\n{best['Description']}\n\n_(Độ khớp: {best_score}%)_"
    return None

# ===== 6. HÀM CHATBOT CHÍNH =====
def chatbot(query: str) -> str:
    if not query or len(remove_vietnamese_diacritics(query)) < 4:
        return "⚠️ Vui lòng nhập câu hỏi rõ hơn."

    # 1) Tra bảng trước (nếu có)
    tb = find_table_by_keyword(query)
    if tb:
        return tb

    # 2) Thử fuzzy môn học
    course_rs = find_course_by_fuzzy_match(query)
    if course_rs:
        return course_rs

    # 3) Vector search từ Sổ tay
    try:
        q_emb = model.encode(query, normalize_embeddings=True).tolist()
        res = col.query(
            query_embeddings=[q_emb],
            include=["documents", "metadatas"],
            where={"source": {"$eq": "So_Tay_Chinh"}},
            n_results=TOP_K
        )
    except Exception as e:
        return f"❌ Lỗi truy vấn: {e}"

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    if not docs:
        return "❌ Không tìm thấy thông tin phù hợp."

    out = ["📘 **Kết quả từ Sổ tay Sinh viên:**\n"]
    for i, d in enumerate(docs):
        if d and d.strip():
            src = (metas[i] or {}).get("source", "So_Tay")
            out.append(f"**Đoạn {i+1}** _(Nguồn: {src})_\n{pretty(d)}\n")
    return "\n".join(out)

# ===== 7. GIAO DIỆN GRADIO =====
demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Nhập câu hỏi", placeholder="VD: Điều kiện nhận học bổng? Thang điểm 4 sang chữ?"),
    outputs="markdown",
    title="🎓 Chatbot Tra cứu HCMUE",
    description="Hỏi về học phần hoặc nội dung Sổ tay. Hệ thống dùng BGE-M3 + ChromaDB để truy vấn."
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)