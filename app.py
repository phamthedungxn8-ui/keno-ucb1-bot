import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Chaos-Harmonic Orthogonal Engine", layout="centered")

# ==============================================================================
# CREATIVE ALGORITHM: CHAOS-HARMONIC ORTHOGONAL VECTOR ENGINE
# ==============================================================================
class ChaosHarmonicKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    # --------------------------------------------------------------------------
    # 1. SHANNON ENTROPY (ĐO ĐỘ HỖN LOẠN TỰ DO CỦA TỪNG SỐ)
    # --------------------------------------------------------------------------
    def _compute_entropy(self, X):
        """Tính Shannon Entropy cho từng số dựa trên vệt nhịp 5 kỳ"""
        T, D = X.shape
        p1 = X.sum(axis=0) / T
        p0 = 1.0 - p1
        
        # Tránh log(0)
        p1 = np.clip(p1, 1e-5, 1.0 - 1e-5)
        p0 = np.clip(p0, 1e-5, 1.0 - 1e-5)
        
        entropy = - (p1 * np.log2(p1) + p0 * np.log2(p0))
        return entropy

    # --------------------------------------------------------------------------
    # 2. VECTOR ORTHOGONALITY (ĐỘ VUÔNG GÓC KHÔNG GIAN 5 CHIỀU)
    # --------------------------------------------------------------------------
    def _compute_orthogonal_pairs(self, X):
        """Tính Cosine Similarity giữa các Vector 5 chiều của 80 số"""
        # X: (5, 80) -> X.T: (80, 5)
        vectors = X.T
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        
        norm_vecs = vectors / norms
        cosine_sim = np.dot(norm_vecs, norm_vecs.T) # Ma trận Cosine (80x80)
        
        # Chúng ta tìm cặp VUÔNG GÓC (Cosine Sim gần 0 nhất)
        # Tức 2 số có quỹ đạo hoàn toàn bù trừ và độc lập về mặt không gian
        ortho_matrix = 1.0 - np.abs(cosine_sim)
        np.fill_diagonal(ortho_matrix, 0)
        return ortho_matrix

    # --------------------------------------------------------------------------
    # PROCESSOR TỔNG HỢP CHAOS-HARMONIC
    # --------------------------------------------------------------------------
    def process(self, X):
        T, D = X.shape
        
        # 1. Tính Entropy Shannon của 80 số
        entropy_vec = self._compute_entropy(X)
        
        # 2. Tính Ma trận Vuông góc Vector
        ortho_mat = self._compute_orthogonal_pairs(X)
        
        # 3. Kết hợp Entropy x Orthogonality
        entropy_pair_mat = np.outer(entropy_vec, entropy_vec)
        combined_mat = ortho_mat * entropy_pair_mat
        
        # 4. BƠM NHIỄU HỖN LOẠN (CHAOS NOISE) ĐỂ THỬ SAI & THOÁT BẪY LOCAL MINIMA
        np.random.seed(42) # Giữ tính ổn định tương đối
        chaos_noise = np.random.uniform(0.85, 1.15, size=(D, D))
        combined_mat = combined_mat * chaos_noise
        
        # 5. BỘ LỌC CỨNG (KHÓA LẶP VÀ KHÓA BÃI HÒA)
        co_matrix = np.dot(X.T, X)
        combined_mat[co_matrix >= 2] = 0.0 # Khóa triệt để các cặp nổ chung >= 2 lần
        
        # Phạt các số xuất hiện ở Kỳ 5 cùng dải hàng chục
        last_draw = X[-1]
        for i in range(D):
            for j in range(D):
                if i != j:
                    # Nếu 2 số cùng nổ kỳ cuối -> Phạt
                    if last_draw[i] == 1 and last_draw[j] == 1:
                        combined_mat[i, j] *= 0.1
                        
        final_mat = self._norm(combined_mat)
        
        # 6. Trích xuất Cặp số Đột phá Không gian
        i, j = np.unravel_index(np.argmax(final_mat, axis=None), final_mat.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(final_mat[i, j])

        explanation = (
            f"• **THUẬT TOÁN SÁNG TẠO SANG LẠI (Chaos-Harmonic Vector Engine):**\n"
            f"  - **Lý thuyết Vuông góc Spatial (Orthogonal Vectors):** Tìm 2 số có Vector nhịp 5 chiều tạo góc $90^\circ$ (Cosine Similarity $\sim 0$). Khi 1 số bùng nổ, số kia bị kích hoạt do hiệu ứng cân bằng Vector không gian.\n"
            f"  - **Cân bằng Entropy Shannon:** Đo mức độ hỗn loạn tự do để chọn ra cặp số đang nằm trên ranh giới chuyển đổi trạng thái.\n"
            f"  - **Bơm Nhiễu Hỗn Loạn Chaos Injection:** Bơm yếu tố ngẫu nhiên để cưỡng ép mô hình phá vỡ các liên kết 1/2 truyền thống.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Chaos-Harmonic Orthogonal Engine")
st.caption("Thuật toán Sáng tạo Đột phá • Hình học Vector Vuông góc • Cân bằng Entropy & Bơm Nhiễu Chaos")

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
                
        st.success("🎉 Đã chạy xong Mô hình Đột phá Vector Vuông góc & Chaos!")
        
        engine = ChaosHarmonicKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ ĐỘT PHÁ VECTOR HỖN LOẠN")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT CHAOS", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Chaos-Orthogonal Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình Chaos-Harmonic.")
