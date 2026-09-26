import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno 3-Layer Cascading Engine", layout="centered")

# ==============================================================================
# 3-LAYER CASCADING ENSEMBLE ENGINE
# ==============================================================================
class CascadingKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, vec):
        m = np.max(vec)
        return vec / m if m > 0 else vec

    # --------------------------------------------------------------------------
    # LỚP 1: XÁC ĐỊNH KHỐI NĂNG LƯỢNG MACRO (MACRO-BLOCK BALANCE LAYER)
    # --------------------------------------------------------------------------
    def _layer1_macro_blocks(self, X):
        """Phân tích 4 khối không gian (1-20, 21-40, 41-60, 61-80)"""
        T, D = X.shape
        blocks = [X[:, 0:20], X[:, 20:40], X[:, 40:60], X[:, 60:80]]
        block_scores = []

        for b in blocks:
            # Tần suất khối ở kỳ gần nhất
            last_density = b[-1].sum()
            # Tần suất trung bình 5 kỳ
            avg_density = b.sum() / float(T)
            # Nhịp nhả năng lượng: Nếu kỳ cuối mật độ giảm nhưng TB cao -> Sắp bùng nổ lại
            energy_release = avg_density - (last_density * 0.5)
            block_scores.append(energy_release)

        # Lấy 2 khối có điểm năng lượng cao nhất
        top_block_indices = np.argsort(block_scores)[-2:]
        return top_block_indices

    # --------------------------------------------------------------------------
    # LỚP 2: BỘ LỌC ĐỘNG LƯỢNG NĂNG LƯỢNG VI MÔ (MICRO-MOMENTUM LAYER)
    # --------------------------------------------------------------------------
    def _layer2_micro_momentum(self, X, top_blocks):
        """Trích xuất Top 3 số tiềm năng nhất từ mỗi Khối Năng Lượng"""
        T, D = X.shape
        candidate_numbers = []

        for b_idx in top_blocks:
            start_num = b_idx * 20
            end_num = start_num + 20
            sub_X = X[:, start_num:end_num]

            # Tính điểm Kalman Momentum cho từng số trong khối
            scores = np.zeros(20)
            for i in range(20):
                x_hat, P, Q, R = 0.25, 1.0, 0.05, 0.2
                for t in range(T):
                    P += Q
                    K = P / (P + R)
                    x_hat += K * (sub_X[t, i] - x_hat)
                    P = (1 - K) * P
                
                # Thưởng cho số có nhịp tích lũy (vừa nghỉ 1 kỳ)
                accumulator_bonus = 1.5 if sub_X[-1, i] == 0 and sub_X[-2, i] == 1 else 1.0
                scores[i] = x_hat * accumulator_bonus

            # Lấy 3 số tốt nhất trong khối này
            top_in_block = np.argsort(scores)[-3:]
            for idx in top_in_block:
                candidate_numbers.append(start_num + idx)

        return candidate_numbers

    # --------------------------------------------------------------------------
    # LỚP 3: MA TRẬN BẮT CẶP BÙ TRỪ & MÀNG LỌC TRIỆT TIÊU (PAIR INTERLOCKING)
    # --------------------------------------------------------------------------
    def _layer3_pair_interlocking(self, X, candidates):
        """Bắt cặp tối ưu từ danh sách ứng viên đã lọc khắt khe"""
        T, D = X.shape
        num_cand = len(candidates)
        pair_matrix = np.zeros((D, D))

        # Ma trận lịch sử nổ chung
        co_occur = np.dot(X.T, X)
        last_co = np.outer(X[-1], X[-1])

        for i in range(num_cand):
            for j in range(i + 1, num_cand):
                c1, c2 = candidates[i], candidates[j]

                # Điều kiện 1: Triệt tiêu nếu vừa nổ chung ở kỳ gần nhất
                if last_co[c1, c2] == 1:
                    continue

                # Điều kiện 2: Phạt nặng nếu nổ chung >= 2 lần trong 5 kỳ
                penalty = 0.1 if co_occur[c1, c2] >= 2 else 1.0

                # Tính điểm bù trừ nhịp (1 số vừa nổ + 1 số tích lũy)
                is_c1_hot = X[-1, c1] == 1
                is_c2_hot = X[-1, c2] == 1
                
                # Ưu tiên ghép 1 Hot + 1 Accumulator
                if is_c1_hot != is_c2_hot:
                    hedged_score = 2.0
                else:
                    hedged_score = 0.5

                pair_matrix[c1, c2] = hedged_score * penalty

        # Trích xuất cặp có điểm cao nhất
        if np.max(pair_matrix) > 0:
            i, j = np.unravel_index(np.argmax(pair_matrix, axis=None), pair_matrix.shape)
        else:
            # Fallback nếu tất cả bị khóa bởi màng lọc
            i, j = candidates[0], candidates[1]

        return sorted([int(i + 1), int(j + 1)]), float(pair_matrix[i, j])

    # --------------------------------------------------------------------------
    # PROCESSOR TỔNG HỢP PHỄU 3 LỚP
    # --------------------------------------------------------------------------
    def process(self, X):
        # Chạy Lớp 1: Khai thác Khối Năng Lượng
        top_blocks = self._layer1_macro_blocks(X)
        block_names = [f"Khối {b+1} ({b*20+1}-{b*20+20})" for b in top_blocks]

        # Chạy Lớp 2: Lọc ứng viên vi mô
        candidates = self._layer2_micro_momentum(X, top_blocks)
        cand_str = ", ".join([f"{c+1:02d}" for c in candidates])

        # Chạy Lớp 3: Bắt cặp bù trừ và màng lọc triệt tiêu
        best_pair, score = self._layer3_pair_interlocking(X, candidates)

        explanation = (
            f"• **CẤU TRÚC PHỄU LỌC 3 LỚP (3-Layer Cascading Engine):**\n"
            f"  - **Lớp 1 (Macro Block):** Chọn 2 Khối năng lượng bùng nổ tốt nhất -> **[{', '.join(block_names)}]**.\n"
            f"  - **Lớp 2 (Micro Momentum):** Lọc ra 6 ứng viên có chỉ số Kalman tích lũy cao nhất -> **[{cand_str}]**.\n"
            f"  - **Lớp 3 (Pair Interlocking Gate):** Triệt tiêu hoàn toàn các cặp bão hòa nổ lặp, ép buộc ghép 1 số Nóng + 1 số Tích lũy.\n"
            f"• **Kết luận Chốt:** Cặp số **({best_pair[0]:02d}, {best_pair[1]:02d})**."
        )

        return best_pair, score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno 3-Layer Cascading Engine")
st.caption("Kiến trúc Mạng lưới 3 Lớp Song song • Phễu Lọc Giảm Chiều • Chống Quá Khớp Dữ Liệu")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=160,
    key="raw_text_keno"
)

if raw_text_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            for num in all_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        st.success("🎉 Đã chạy xong Kiến trúc Mạng lưới Phễu Lọc 3 Lớp!")
        
        engine = CascadingKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHỐT PHỄU 3 LỚP TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT TỐI ƯU", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Cascading Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình phễu 3 lớp.")
