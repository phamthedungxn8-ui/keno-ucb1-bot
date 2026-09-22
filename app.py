import streamlit as st
import numpy as np
import pandas as pd

# ==========================================
# 1. ĐỊNH NGHĨA CLASS ENGINE (ĐẶT ĐẦU FILE)
# ==========================================
class KenoMemoryPairEngineOptimized:
    """
    Bộ trích xuất Cặp Bậc 2 Keno nâng cấp:
    Tích hợp Hàm Phạt Bão Hòa (Penalized Saturation) chống bẫy quá nhiệt.
    """
    def __init__(self, window_size: int = 5, decay_lambda: float = 0.25, gamma_sat: float = 3.0):
        self.W = window_size
        self.weights = np.exp(-decay_lambda * np.arange(window_size))[::-1] # Ưu tiên kỳ gần nhất
        self.gamma_sat = gamma_sat

    def extract_best_pair(self, history_matrix: np.ndarray, renyi_s: float = 0.0) -> tuple[tuple[int, int], float, dict]:
        """
        Trích xuất cặp số bậc 2 tối ưu dựa trên lịch sử N kỳ gần nhất.
        """
        T, N = history_matrix.shape
        if T < self.W:
            return (1, 2), 0.0, {"Cảnh báo": f"Chưa đủ {self.W} kỳ lịch sử để phân tích"}

        # Cửa sổ W kỳ gần nhất (VD: 5 x 80)
        recent = history_matrix[-self.W:] 
        
        # 1. Tần suất xuất hiện thô trong cửa sổ
        counts = np.sum(recent, axis=0) # Mảng 80 phần tử
        
        # 2. Phương trình 1: Ký nhớ có Phạt Bão Hòa (M_i)
        raw_m = np.dot(self.weights, recent) # Tích chập trọng số thời gian
        sat_penalty = 1.0 - (counts / float(self.W)) ** self.gamma_sat
        m_t = raw_m * sat_penalty # Kìm hãm các số về quá nhiều (>=4 lần)

        # 3. Phương trình 2: Ma trận Tương quan Cặp (C_ij)
        co_occurrence = np.dot(recent.T, recent)
        deg = np.diag(co_occurrence)
        norm_factor = np.sqrt(np.outer(deg, deg)) + 1e-9
        c_t = co_occurrence / norm_factor
        np.fill_diagonal(c_t, 0.0)

        # 4. Phương trình 3: Điểm Cộng hưởng Cặp (S_ij)
        m_matrix = np.add.outer(m_t, m_t)
        np.fill_diagonal(m_matrix, 0.0)
        
        # Tỷ lệ: 60% Năng lượng đơn + 40% Tương quan cặp (kèm bù Entropy Rényi)
        score_matrix = (0.6 * m_matrix + 0.4 * c_t) * np.exp(-renyi_s)
        
        # Tìm chỉ số Cặp có điểm cao nhất
        best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        num1, num2 = int(best_idx[0] + 1), int(best_idx[1] + 1)
        
        stats = {
            f"Số {num1:02d}": f"Về {int(counts[best_idx[0]])}/{self.W} kỳ | Điểm M_i: {m_t[best_idx[0]]:.3f}",
            f"Số {num2:02d}": f"Về {int(counts[best_idx[1]])}/{self.W} kỳ | Điểm M_i: {m_t[best_idx[1]]:.3f}",
            "Chỉ số Tương quan C_ij": f"{c_t[best_idx]:.3f}",
            "Tổng điểm S_ij": f"{score_matrix[best_idx]:.4f}"
        }
        return (num1, num2), float(score_matrix[best_idx]), stats


# ==========================================
# 2. KHỞI TẠO CACHE RESOURCE & SESSION STATE
# ==========================================
@st.cache_resource
def get_pair_engine():
    """Hàm khởi tạo Engine an toàn tránh NameError và giảm tải rerun."""
    return KenoMemoryPairEngineOptimized(window_size=5, decay_lambda=0.25, gamma_sat=3.0)

# Khởi tạo instance
pair_engine = get_pair_engine()

if "pair_engine" not in st.session_state:
    st.session_state.pair_engine = pair_engine


# ==========================================
# 3. GIAO DIỆN UNG DỤNG STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="Keno AI Engine - Bậc 2 Tối Ưu", layout="wide")
    st.title("🎯 Phân Tích & Tối Ưu Cặp Bậc 2 Keno")

    # Dữ liệu giả lập 5 kỳ mẫu để kiểm thử (hoặc thay bằng dữ liệu thực từ DB/API)
    st.subheader("📋 Lịch sử 5 kỳ gần nhất")
    
    # Ma trận nhị phân 5x80 (ví dụ)
    sample_data = np.zeros((5, 80), dtype=int)
    # Kỳ 1: #296683
    sample_data[0, [3, 5, 15, 16, 17, 18, 21, 26, 39, 41, 44, 45, 46, 47, 50, 53, 61, 64, 68, 76]] = 1
    # Kỳ 2: #296684
    sample_data[1, [1, 9, 13, 15, 17, 20, 21, 22, 23, 24, 25, 39, 42, 45, 46, 50, 51, 59, 73, 77]] = 1
    # Kỳ 3: #296685
    sample_data[2, [1, 2, 4, 6, 8, 10, 11, 14, 39, 44, 46, 48, 49, 50, 51, 58, 66, 68, 69, 79]] = 1
    # Kỳ 4: #296686
    sample_data[3, [2, 5, 10, 14, 15, 18, 19, 21, 25, 27, 33, 37, 39, 48, 56, 64, 65, 67, 72, 75]] = 1
    # Kỳ 5: #296687
    sample_data[4, [7, 17, 19, 20, 21, 22, 24, 27, 30, 31, 37, 47, 53, 54, 56, 61, 74, 76, 77, 79]] = 1

    st.write("Đã nạp ma trận 5 kỳ quay lịch sử (#296683 - #296687).")

    if st.button("🚀 Chạy Phân Tích & Dự Báo Cặp Tối Ưu", type="primary"):
        # Gọi engine từ session_state
        best_pair, score, stats = st.session_state.pair_engine.extract_best_pair(sample_data)

        st.success(f"🔥 Cặp Bậc 2 Tối Ưu Nhất Cho Kỳ Tiếp Theo: **({best_pair[0]:02d}, {best_pair[1]:02d})**")
        
        st.markdown("### 📊 Thống kê chi tiết thuật toán")
        col1, col2 = st.columns(2)
        with col1:
            for k, v in list(stats.items())[:2]:
                st.metric(label=k, value=v)
        with col2:
            for k, v in list(stats.items())[2:]:
                st.metric(label=k, value=v)

if __name__ == "__main__":
    main()
