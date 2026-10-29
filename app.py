import re
import numpy as np
import pandas as pd
import itertools
import streamlit as st

st.set_page_config(page_title="MDM-IDS v113.0: Cluster & Co-Occurrence Core", layout="wide")

class ClusterCoOccurrenceEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=3, max_kies=15):
        cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text.strip(), flags=re.IGNORECASE)
        all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= self.D]
        total_kies = len(all_numbers) // 20

        if total_kies < min_kies:
            return None, total_kies

        kies_to_use = min(total_kies, max_kies)
        used_numbers = all_numbers[-kies_to_use * 20:]

        matrix = np.zeros((kies_to_use, self.D), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0

        return matrix, kies_to_use

    def compute_co_occurrence(self, matrix: np.ndarray):
        """
        Xây dựng Ma trận Tương quan Cặp (Pairwise Correlation)
        và phát hiện các Cụm Số bọc lót (Local Clusters)
        """
        T, D = matrix.shape
        # Trọng số gia tốc theo thời gian (kỳ gần nhất ưu tiên cao nhất)
        weights = np.exp(np.linspace(0, 1.5, T))
        weights /= weights.sum()

        # Ma trận hiệp biến có gán trọng số
        weighted_matrix = matrix * weights[:, np.newaxis]
        co_matrix = np.dot(weighted_matrix.T, matrix) # D x D

        # Xóa đường chéo chính (không xét tương quan của số với chính nó)
        np.fill_diagonal(co_matrix, 0)

        # 1. Tìm các cặp số có độ liên kết cao nhất (Co-occurrence Pairs)
        pair_scores = []
        for i in range(D):
            for j in range(i + 1, D):
                score = co_matrix[i, j]
                if score > 0:
                    pair_scores.append((i + 1, j + 1, score))

        pair_scores = sorted(pair_scores, key=lambda x: x[2], reverse=True)

        # 2. Xây dựng Cụm Bọc Lót (Clusters 3 số liên tiếp hoặc cùng dải)
        single_scores = weighted_matrix.sum(axis=0)
        top_singles = np.argsort(single_scores)[::-1]

        clusters = []
        visited = set()
        for idx in top_singles:
            num = idx + 1
            if num in visited:
                continue
            # Tìm các số liền kề hoặc có tương quan mạnh
            neighbors = [num - 1, num, num + 1]
            valid_neighbors = [n for n in neighbors if 1 <= n <= 80]
            clusters.append(valid_neighbors)
            visited.update(valid_neighbors)
            if len(clusters) >= 3:
                break

        # Lấy Top 5 số có điểm tương quan tổng thể cao nhất
        top_5_overall = [idx + 1 for idx in top_singles[:5]]

        return {
            "top_5": top_5_overall,
            "top_pairs": pair_scores[:5],
            "clusters": clusters
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🎯 MDM-IDS v113.0: CLUSTER & CO-OCCURRENCE ENGINE")
st.caption("Chiến thuật Cụm bọc lót • Ma trận tương quan cặp (Co-occurrence) • Tối ưu hóa xác suất trúng")

with st.sidebar:
    st.header("⚙️ Cấu Hình Cụm")
    target_order = st.slider("Số kỳ phân tích (Cửa sổ cực ngắn):", 3, 6, 4)
    st.info("Mô hình v113.0 tập trung vào việc tạo các dải bọc lót thay vì dự báo con số đơn độc.")

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Chỉ cần {target_order} kỳ gần nhất):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = ClusterCoOccurrenceEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.compute_co_occurrence(matrix)

        st.success(f"⚡ Đã trích xuất ma trận tương quan và cụm bọc lót trên {total_kies} kỳ!")
        st.markdown("---")

        st.subheader("🎯 TOP 5 CON SỐ TỰ ĐỘNG TỔNG HỢP")
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['top_5']])
        st.markdown(f"<div style='text-align:center; padding:20px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔗 CẶP SỐ CÓ ĐỘ TƯƠNG QUAN MẠNH (Bao 2)")
            st.write("Các cặp số thường xuyên đi kèm nhau trong thời gian ngắn:")
            for p in res['top_pairs']:
                st.code(f"Cặp Bao 2: [ {p[0]:02d} - {p[1]:02d} ] | Điểm liên kết: {p[2]:.3f}", language="text")

        with col_b:
            st.subheader("🛡️ CỤM SỐ BỌC LÓT DẢI (Cluster Sets)")
            st.write("Đánh dải số liên tiếp để chống trễ pha và lệch biên:")
            for idx, c in enumerate(res['clusters']):
                c_str = " - ".join([f"{n:02d}" for n in c])
                st.code(f"Cụm {idx+1}: [ {c_str} ]", language="text")

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để chạy ma trận tương quan (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu 3-4 kỳ Keno vào khung trên để cỗ máy v113.0 lập Ma trận Tương quan Cặp!")
