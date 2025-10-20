from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from fuzzywuzzy import fuzz
from unidecode import unidecode
import pandas as pd
import json, os, re, sys

# 🌟 NHẬP THƯ VIỆN GRADIO
import gradio as gr 


# ========================================
# 1️⃣ KHỞI TẠO (GIỮ NGUYÊN)
# ========================================
# Ứng dụng này sẽ được khởi chạy bởi Gradio, không cần đối tượng FastAPI

# ========================================
# 2️⃣ HÀM HỖ TRỢ (GIỮ NGUYÊN)
# ========================================

def remove_vietnamese_diacritics(text: str) -> str:
    """Loại bỏ dấu tiếng Việt và chuyển sang chữ thường."""
    return unidecode(text).lower()

def pretty(text: str) -> str:
    """Định dạng văn bản dễ đọc hơn."""
    text = re.sub(r"([;:.\)\]\}])\s*(?=[\+\-•–])", r"\1\n", text)
    text = re.sub(r"\s*([+\-•–])\s+", r"\n\1 ", text)
    text = re.sub(r"\s*(Điều\s+\d+\.)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def classify_query_intent(query: str) -> str:
    """Phân loại truy vấn là về Môn học hay Sổ tay."""
    normalized_query = remove_vietnamese_diacritics(query)

    course_phrases = [
        "lap trinh co ban", "co so toan", "toan roi rac", "thiet ke web",
        "duong loi quoc phong", "phap luat dai cuong", "triet hoc mac lenin",
        "tam ly hoc", "giao duc the chat", "lap trinh nang cao",
        "lap trinh huong doi tuong", "cong tac quoc phong", "kinh te chinh tri",
        "chu nghia xa hoi", "phuong phap nghien cuu khoa hoc", "giao duc doi song",
        "phuong phap hoc tap", "ky nang thich ung", "ky nang lam viec nhom",
        "cau truc du lieu", "co so du lieu", "lap trinh tren windows",
        "xac suat thong ke", "ly thuyet do thi", "quan su chung",
        "tu tuong ho chi minh", "kien truc may tinh", "nhap mon mang may tinh",
        "he dieu hanh", "phan tich va thiet ke giai thuat", "quy hoach tuyen tinh",
        "ky thuat chien dau", "lich su dang cong san", "nhap mon cong nghe phan mem",
        "phan tich thiet ke huong doi tuong", "tri tue nhan tao", "cac he co so du lieu",
        "thiet ke va quan ly mang lan", "phan tich va thiet ke he thong thong tin",
        "co so du lieu nang cao", "he thong ma nguon mo", "xu ly anh so",
        "quan tri co ban voi windows server", "nghi thuc giao tiep mang",
        "phat trien ung dung tren thiet bi di dong", "quan ly du an cong nghe thong tin",
        "kiem thu phan mem", "phat trien ung dung tro cho choi",
        "quy trinh phat trien phan mem agile", "he thong nhung", "hoc may",
        "lap trinh python", "lap trinh php", "thuc hanh nghe nghiep",
        "mang may tinh nang cao", "cong nghe web", "cong nghe java",
        "cac he co so tri thuc", "do hoa may tinh", "bao mat va an ninh mang",
        "logic mo", "cong nghe net", "chuyen de oracle", "truyen thong ky thuat so",
        "chuan doan va quan ly su co mang", "dinh tuyen mang nang cao",
        "quan tri mang voi linux", "quan tri dich vu mang",
        "he thong quan tri doanh nghiep", "xay dung du an cong nghe thong tin",
        "he tu van thong tin", "bao mat co so du lieu", "khai thac du lieu va ung dung",
        "lap trinh tien hoa", "cac phuong phap hoc thong ke",
        "lap rap cai dat va bao tri may tinh", "internet van vat", "nhap mon devops",
        "cong nghe chuoi khoi", "cac giai thuat tinh toan dai so",
        "khai thac du lieu van ban", "xu ly ngon ngu tu nhien", "ly thuyet ma hoa va mat ma",
        "thuc tap nghe nghiep", "khoi nghiep", "cong nghe phan mem nang cao",
        "cong nghe mang khong day", "thuong mai dien tu", "kiem thu phan mem nang cao",
        "dien toan dam may", "do hoa may tinh nang cao", "phan tich du lieu",
        "may hoc nang cao", "thi giac may tinh", "phan tich anh y khoa",
        "phat trien ung dung tren thiet bi di dong nang cao", "khoa luan tot nghiep",
        "ho so tot nghiep", "san pham nghien cuu",
        # Thêm các từ khóa mô tả ý định
        "mo ta mon", "thong tin mon", "hoc phan", "mon hoc", "hoc gi"
    ]

    strong_single_keywords = [
        "mon", "hoc phan", "lap trinh", "toan", "python", "java", "web",
        "du lieu", "ai", "linux", "windows", "thong ke", "do hoa", "bao mat",
        "mang", "phap luat", "triet hoc", "tam ly", "lich su", "kinh te",
        "phat trien", "thiet ke", "quantri", "server", "oracle", "devops",
        "agile", "khoi nghiep", "vat ly", "hoa hoc", "su pham", "tin hoc",
        "huong doi tuong"
    ]

    if any(phrase in normalized_query for phrase in course_phrases):
        return "COURSE"

    if len(normalized_query.split()) <= 4 and any(k in normalized_query for k in strong_single_keywords):
        return "COURSE"

    return "GENERAL"

# ========================================
# 3️⃣ CẤU HÌNH & TẢI MÔ HÌNH (GIỮ NGUYÊN)
# ========================================
DB_DIR = "./vector_store"
COLLECTION = "so_tay_hcmue"
TABLE_JSON = "./so_tay_all_tables_clean.json"
COURSE_JSON = "./mon_hoc_mo_ta.json"
MODEL_NAME = "BAAI/bge-m3"

print("🚀 Đang tải mô hình...")
model = SentenceTransformer(MODEL_NAME)

chromadb.api.client.SharedSystemClient._instance = None
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings())
col = None

if COLLECTION in [c.name for c in client.list_collections()]:
    col = client.get_collection(COLLECTION)
else:
    col = client.create_collection(COLLECTION)

# ========================================
# 4️⃣ TẢI DỮ LIỆU JSON (GIỮ NGUYÊN)
# ========================================
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

tables = load_json(TABLE_JSON)
courses = load_json(COURSE_JSON)
COURSE_DATA = {
    remove_vietnamese_diacritics(c["ten_mon"]): {
        "ten_mon": c["ten_mon"],
        "Description": c["Description"]
    }
    for c in courses
}

# ========================================
# 5️⃣ TRA CỨU BẢNG / MÔN HỌC / VECTOR (GIỮ NGUYÊN)
# ========================================

def find_table_by_keyword(query: str):
    normalized_query = remove_vietnamese_diacritics(query)
    mapping = {
        "4 sang chu": ["thang_diem_4"],
        "hoc bong": ["xep_loai_hoc_bong"],
        "chu sang 10": ["thang_diem_10_chu"],
        "xep loai hoc luc": ["xep_loai_hoc_luc"],
        "yeu cau hoc bong": ["yeu_cau_hoc_bong"],
        "diem ren luyen": [
            "diem_ren_luyen1", "diem_ren_luyen2",
            "diem_ren_luyen3", "diem_ren_luyen4", "diem_ren_luyen5"
        ]
    }

    best_key, best_score = None, 0
    for k in mapping.keys():
        score = max(
            fuzz.WRatio(normalized_query, k),
            fuzz.partial_ratio(normalized_query, k),
            fuzz.token_set_ratio(normalized_query, k)
        )
        if score > best_score and score >= 85:
            best_key, best_score = k, score

    if not best_key:
        return None

    out = [f"### 📊 Kết quả tra cứu Bảng (Độ khớp: {best_score}%)"]
    for type_name in mapping[best_key]:
        for t in tables:
            if t.get("type") == type_name:
                df = pd.DataFrame(t["data"])
                title = t.get("title", type_name.replace("_", " ").title())
                # Sử dụng to_html để hiển thị đẹp hơn trong Gradio
                out.append(f"#### {title}\n{df.to_html(index=False)}") 
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
    if best and best_score >= 85:
        return f"📚 **Môn học:** {best['ten_mon']}\n\n{best['Description']}\n_(Độ khớp: {best_score}%)_"
    return None


def vector_search(query: str, top_k=5):
    try:
        q_emb = model.encode(query, normalize_embeddings=True).tolist()
        res = col.query(
            query_embeddings=[q_emb],
            include=["documents", "metadatas"],
            where={"source": {"$eq": "So_Tay_Chinh"}},
            n_results=top_k
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out = []
        for i, d in enumerate(docs):
            if d and d.strip():
                src = (metas[i] or {}).get("source", "So_Tay")
                out.append(f"**Đoạn {i+1}** _(Nguồn: {src})_\n{pretty(d)}\n")
        return "\n".join(out) if out else "❌ Không tìm thấy thông tin phù hợp."
    except Exception as e:
        return f"❌ Lỗi truy vấn vector: {e}"

# ========================================
# 6️⃣ HÀM XỬ LÝ CHÍNH CHO GRADIO (thay thế route /query)
# ========================================

def chatbot_query(question: str) -> str:
    """Hàm xử lý logic truy vấn, trả về chuỗi kết quả."""
    question = question.strip()

    if not question:
        return "Vui lòng nhập câu hỏi."
    
    # In ra để tiện debug trong console (nếu chạy local)
    intent = classify_query_intent(question)
    print(f"🧠 Phân loại truy vấn: {intent}")

    # 1️⃣ Truy vấn bảng
    tb = find_table_by_keyword(question)
    if tb:
        return tb

    # 2️⃣ Môn học
    if intent == "COURSE":
        course = find_course_by_fuzzy_match(question)
        if course:
            return course

    # 3️⃣ Mặc định → vector search
    vec = vector_search(question)
    return vec

# ========================================
# 7️⃣ GIAO DIỆN VÀ CHẠY APP GRADIO (thay thế Uvicorn)
# ========================================

# Khởi tạo giao diện Gradio
gr_interface = gr.Interface(
    # Hàm xử lý chính
    fn=chatbot_query,
    # Đầu vào là một ô văn bản (Text input)
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Nhập câu hỏi của bạn (ví dụ: 'diem ren luyen co may loai', 'mo ta mon lap trinh python', 'tieu chi xep loai hoc luc')", 
        label="Câu hỏi tra cứu"
    ),
    # Đầu ra là một ô văn bản hiển thị (Output text, có hỗ trợ Markdown)
    outputs=gr.Markdown(
        label="Kết quả tra cứu"
    ),
    # Tiêu đề và mô tả
    title="🎓 Chatbot HCMUE Demo (Gradio)",
    description="API tra cứu Sổ tay Sinh viên & Môn học HCMUE, được chuyển đổi sang giao diện Gradio.",
    
    # Ví dụ để người dùng dễ thử nghiệm
    examples=[
        "Bảng quy đổi điểm từ thang 4 sang điểm chữ",
        "mô tả môn học cơ sở dữ liệu",
        "tiêu chí xếp loại học lực",
        "tôi cần biết về điểm rèn luyện"
    ],
    allow_flagging='never', # Tắt tính năng gắn cờ
    theme=gr.themes.Soft() # Chọn một theme nhẹ nhàng hơn
)

# Chạy giao diện
if __name__ == "__main__":
    gr_interface.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860)) # Gradio thường dùng port 7860
    )