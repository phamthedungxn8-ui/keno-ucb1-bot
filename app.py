import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Multi-Agent Engine", layout="centered")

# ==============================================================================
# 1. HEURISTIC & MATHEMATICAL AGENTS
# ==============================================================================
class AdvancedMetaAgent:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def process(self, matrix):
        """
        matrix: numpy array shape (5, 80)
        """
        T, D = matrix.shape
        freqs = matrix.sum(axis=0)
        hot_indices = np.where(freqs >= 2)[0]
        
        # ----------------------------------------------------------------------
        # THUẬT TOÁN 1: Entanglement Impulse (Lực kéo hạt nhân lặp)
        # ----------------------------------------------------------------------
        entangle_mat = np.zeros((D, D))
        for t in range(T):
            active = np.where(matrix[t] == 1)[0]
            for i in active:
                for j in active:
                    if i != j:
                        w = 1.0
                        if i in hot_indices: w += 1.5
                        if j in hot_indices: w += 1.5
                        entangle_mat[i, j] += w
        np.fill_diagonal(entangle_mat, 0)

        # ----------------------------------------------------------------------
        # THUẬT TOÁN 2: Holographic Distance Graph (Mạng khoảng cách toàn ảnh)
        # ----------------------------------------------------------------------
        holo_mat = np.zeros((D, D))
        for t in range(T):
            active = np.where(matrix[t] == 1)[0]
            # Tính hiệu số khoảng cách giữa các con số
            diffs = [abs(active[a] - active[b]) for a in range(len(active)) for b in range(a+1, len(active))]
            common_diff = max(set(diffs), key=diffs.count) if diffs else 0
            
            for i in active:
                for j in active:
                    if i != j and abs(i - j) == common_diff:
                        holo_mat[i, j] += 2.0
        np.fill_diagonal(holo_mat, 0)

        # ----------------------------------------------------------------------
        # THUẬT TOÁN 3: SVD Low-Rank Noise Reduction (Phân rã lọc nhiễu SVD)
        # ----------------------------------------------------------------------
        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            # Chỉ giữ lại thành phần tín hiệu chủ đạo (Rank-1/Rank-2)
            S_clean = np.zeros_like(S)
            S_clean[0] = S[0]
            if len(S) > 1: S_clean[1] = S[1]
            clean_matrix = np.dot(U, np.dot(np.diag(S_clean), Vt))
            svd_mat = np.dot(clean_matrix.T, clean_matrix)
            np.fill_diagonal(svd_mat, 0)
        except DistributionError:
            svd_mat = np.zeros((D, D))

        # ----------------------------------------------------------------------
        # TỔNG HỢP CỘNG HƯỞNG (Consensus Matrix)
        # ----------------------------------------------------------------------
        final_operator = entangle_mat + 1.2 * holo_mat + 1.5 * svd_mat
        np.fill_diagonal(final_operator, 0)

        i, j = np.unravel_index(np.argmax(final_operator, axis=None), final_operator.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        score = float(final_operator[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in hot_indices]) if len(hot_indices) > 0 else "Không có"
        explanation = (
            f"Hạt nhân lặp: [{hot_str}] | "
            f"Hệ thống Multi-Agent (Entanglement + Holographic Graph + SVD) đã đồng thuận chọn cặp ({num1:02d}, {num2:02d})."
        )

        return (num1, num2), score, explanation

# ==============================================================================
# 2. STREAMLIT UI
# ==============================================================================
st.title("⚡ Multi-Agent Keno Engine")
st.caption("Tích hợp SVD Lọc nhiễu • Holographic Graph • Lực kéo Entanglement")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 04 06 10 18 ...\nKì 2: ...",
    height=160,
    key="raw_text_keno"
)

if raw_text_input.strip():
    raw_data = raw_text_input.strip()
    
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_data, flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        valid_numbers = all_numbers[:100]
        
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            ky_nums = valid_numbers[k * 20 : (k + 1) * 20]
            for num in ky_nums:
                matrix[k, num - 1] = 1.0
                
        st.success("🎉 Multi-Agent đã quét và phân rã thành công 5 kỳ lịch sử!")
        
        agent = AdvancedMetaAgent(num_dim=80)
        best_pair, score, explanation = agent.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 ĐỒNG THUẬN MULTI-AGENT (BẬC 2)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="CẶP SỐ TỐI ƯU CHỐT",
                value=f"{best_pair[0]:02d} — {best_pair[1]:02d}"
            )
        with col2:
            st.metric(
                label="Chỉ Số Đồng Thuận (Consensus)",
                value=f"{score:.2f}"
            )
            
        st.info(f"🧠 **Phán đoán tổng hợp:** {explanation}")
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Hãy dán chuỗi số 5 kỳ vào khung phía trên để hệ thống Multi-Agent tự động kích hoạt.")
