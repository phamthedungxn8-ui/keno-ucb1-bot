import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MDM-IDS v20.0 Hyper-Dimensional Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# LỚP 0: UNIFIED HYPER-TENSOR PIPELINE (MÔ HÌNH DỮ LIỆU ĐA TẦN SỐ)
# ==============================================================================
class HyperTensorPipeline:
    """Quản lý, chuẩn hóa và phân tích Tensor dữ liệu 2D/3D đa khung thời gian (10-30 kỳ)."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def build_tensor(self, raw_text: str, min_kies=10, max_kies=30):
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


# ==============================================================================
# LỚP 1: CONTINUOUS ENTROPY & RIEMANN PHASE DYNAMICS
# ==============================================================================
class RiemannPhaseEntropyLayer:
    """Đo đạc Entropy, ma trận Covariance SVD và độ bù lệch pha nhị nguyên."""
    def process(self, X):
        T, D = X.shape
        
        # 1. Trọng số suy giảm mũ theo thời gian (Exponential Time-Decay)
        decay_weights = np.exp(np.linspace(-1.2, 0, T))
        decay_weights /= decay_weights.sum()
        weighted_freqs = np.dot(decay_weights, X)

        # 2. Shannon Entropy trên từng nút số
        p1 = np.mean(X == 1, axis=0) + 1e-9
        p0 = np.mean(X == 0, axis=0) + 1e-9
        entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))

        # 3. Ma trận vướng víu cặp (Co-occurrence Covariance) trên cửa sổ 15 kỳ gần nhất
        X_med = X[-min(15, T):]
        cov_matrix = np.corrcoef(X_med.T)
        cov_matrix = np.nan_to_num(cov_matrix)
        pair_density = np.sum(np.maximum(0, cov_matrix), axis=1) / float(D)

        # 4. Bù lệch pha Nhị nguyên (Duality Compensation)
        recent_k = X[-1]
        even_mask = np.array([1 if (i + 1) % 2 == 0 else 0 for i in range(D)])
        large_mask = np.array([1 if (i + 1) > 40 else 0 for i in range(D)])

        even_ratio = np.sum(recent_k * even_mask) / 20.0
        large_ratio = np.sum(recent_k * large_mask) / 20.0

        duality_bias = np.zeros(D)
        for i in range(D):
            b_even = (1.0 - even_ratio) if even_mask[i] else even_ratio
            b_large = (1.0 - large_ratio) if large_mask[i] else large_ratio
            duality_bias[i] = 0.5 * (b_even + b_large)

        return weighted_freqs, entropy, pair_density, duality_bias


# ==============================================================================
# LỚP 2: MULTI-AGENT QUANTUM NASH CONSENSUS ENGINE
# ==============================================================================
class MultiAgentNashEngine:
    """Hợp nhất các Agent suy luận độc lập theo nguyên lý Cân bằng Nash."""
    def process(self, X, weighted_freqs, entropy, pair_density, duality_bias):
        T, D = X.shape
        freqs_short = X[-min(6, T):].sum(axis=0)

        # Agent 1: Xung lực ngắn (Micro-Impulse)
        agent1_score = weighted_freqs

        # Agent 2: Ranh giới Pha bộc phát (Phase-Transition Boundary)
        agent2_score = np.ones(D)
        for i in range(D):
            if freqs_short[i] in [1, 2]:
                agent2_score[i] = 2.0  # Tối ưu điểm bùng nổ
            elif freqs_short[i] == 0:
                agent2_score[i] = 0.30 # Phạt số đóng băng
            elif freqs_short[i] >= 4:
                agent2_score[i] = 0.10 # Phạt số bão hòa

        # Agent 3: Mật độ vướng víu macro
        agent3_score = pair_density * entropy

        # Chuẩn hóa Min-Max các Agent
        norm1 = (agent1_score - agent1_score.min()) / (agent1_score.max() - agent1_score.min() + 1e-9)
        norm3 = (agent3_score - agent3_score.min()) / (agent3_score.max() - agent3_score.min() + 1e-9)
        norm_dual = (duality_bias - duality_bias.min()) / (duality_bias.max() - duality_bias.min() + 1e-9)

        # Tích chéo hội tụ (Nash Equilibrium Convergence)
        consensus_vector = (norm1 ** 1.0) * (norm3 ** 1.4) * (norm_dual ** 1.2) * agent2_score
        return consensus_vector


# ==============================================================================
# LỚP 3: PHASE-INVERSION & REPETITION SUPPRESSION FILTER
# ==============================================================================
class SuppressionFilterLayer:
    """Triệt tiêu hiện tượng kẹt bão hòa thanh ghi và lặp số tức thời T-1."""
    def filter(self, X, consensus_vector):
        D = X.shape[1]
        final_scores = consensus_vector.copy()

        # Phạt triệt tiêu lặp số ở kỳ T-1
        for i in range(D):
            if X[-1, i] == 1:
                final_scores[i] *= 0.18 # Phạt mạnh số vừa ra ở kỳ trước

        return final_scores


# ==============================================================================
# LỚP 4: UNIVERSAL MULTI-TIER COMBINATORIAL ROUTER
# ==============================================================================
class UniversalCombinatorialRouter:
    """Tự động phân rã chỉ số năng lượng thành các bộ số tối ưu cho TẮT CẢ BẬC CHƠI."""
    def route_all_tiers(self, final_scores):
        D = len(final_scores)
        ranked_indices = np.argsort(final_scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]

        # 1. Cấu trúc Dual-Core Phân bổ Lực
        alpha_core = [ranked_numbers[i] for i in range(0, 16, 2)] # Các nút vị trí lẻ (Top 1, 3, 5...)
        beta_core = [ranked_numbers[i] for i in range(1, 16, 2)]  # Các nút vị trí chẵn (Top 2, 4, 6...)

        # 2. Định tuyến cho Bậc Nhỏ (Bậc 2, Bậc 3, Bậc 4) - Yêu cầu độ chuẩn cặp
        bac2_alpha = sorted(alpha_core[:2])
        bac2_beta = sorted(beta_core[:2])
        bac2_backup = sorted([ranked_numbers[16], ranked_numbers[17]])

        bac3_alpha = sorted(alpha_core[:3])
        bac3_beta = sorted(beta_core[:3])

        bac4_alpha = sorted(alpha_core[:4])
        bac4_beta = sorted(beta_core[:4])

        # 3. Định tuyến cho Bậc Lớn (Bậc 7, Bậc 8, Bậc 9) - Yêu cầu bao phủ diện rộng & bảo hiểm
        bac7_master = sorted(ranked_numbers[:7])
        bac8_master = sorted(ranked_numbers[:8])
        bac9_master = sorted(ranked_numbers[:9])
        bac8_backup = sorted(ranked_numbers[8:16])

        return {
            "ranked_all": ranked_numbers,
            "scores": final_scores,
            "bac2_alpha": bac2_alpha,
            "bac2_beta": bac2_beta,
            "bac2_backup": bac2_backup,
            "bac3_alpha": bac3_alpha,
            "bac3_beta": bac3_beta,
            "bac4_alpha": bac4_alpha,
            "bac4_beta": bac4_beta,
            "bac7_master": bac7_master,
            "bac8_master": bac8_master,
            "bac9_master": bac9_master,
            "bac8_backup": bac8_backup
        }


# ==============================================================================
# STREAMLIT UI SYSTEM (HD-PME v20.0)
# ==============================================================================
st.title("🌌 MDM-IDS v20.0: HYPER-DIMENSIONAL ENGINE")
st.caption("Siêu Cấu Trúc Dự Đoán Keno Đa Bậc • 5 Lớp Thuật Toán Độc Lập • Cân Bằng Nash & Phân Tầng Pha")

with st.sidebar:
    st.header("⚙️ Cấu Hình Siêu Hệ Thống")
    min_window = st.slider("Cửa sổ tối thiểu (Kỳ):", 6, 15, 10)
    max_window = st.slider("Cửa sổ tối đa (Kỳ):", 15, 50, 30)
    st.info("Hệ thống tự động điều chỉnh ma trận trọng số Decay theo kích thước cửa sổ nhập vào.")

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Khuyên dùng từ 10 - 30 kỳ để đạt độ ổn định tối đa):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...\n...\nKỳ 20: ...",
    height=180
)

if raw_input.strip():
    pipeline = HyperTensorPipeline()
    matrix, total_kies = pipeline.build_tensor(raw_input, min_kies=min_window, max_kies=max_window)

    if matrix is not None:
        # Thực thi 5 Lớp Pipeline
        l1 = RiemannPhaseEntropyLayer()
        w_freqs, entropy, pair_density, duality = l1.process(matrix)

        l2 = MultiAgentNashEngine()
        consensus = l2.process(matrix, w_freqs, entropy, pair_density, duality)

        l3 = SuppressionFilterLayer()
        final_scores = l3.filter(matrix, consensus)

        l4 = UniversalCombinatorialRouter()
        res = l4.route_all_tiers(final_scores)

        st.success(f"⚡ Đã thực thi hoàn tất 5 Lớp Siêu Cấu Trúc HD-PME v20.0 trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        # TẠO TABS CHO CÁC NHÓM BẬC CỤ THỂ
        tab_small, tab_large, tab_analytics = st.tabs([
            "🎯 NHÓM BẬC NHỎ (BẬC 2, 3, 4)", 
            "🛡️ NHÓM BẬC LỚN (BẬC 7, 8, 9)", 
            "📊 PHÂN TÍCH MA TRẬN NĂNG LƯỢNG"
        ])

        # TAB 1: BẬC NHỎ
        with tab_small:
            st.subheader("🔥 BẬC 2 (TỐI ƯU DÀN CẶP - THƯỞNG 90.000 VNĐ)")
            c2a, c2b, c2c = st.columns(3)
            with c2a:
                st.markdown(f"<div style='text-align:center; padding:15px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                            f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA</span>"
                            f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h2>"
                            f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Dàn Tích Lũy Pha</p></div>", unsafe_allow_html=True)
            with c2b:
                st.markdown(f"<div style='text-align:center; padding:15px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                            f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (BÙNG NỔ)</span>"
                            f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h2>"
                            f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Dàn Phụ Bộc Phát</p></div>", unsafe_allow_html=True)
            with c2c:
                st.markdown(f"<div style='text-align:center; padding:15px; background:#111; border-radius:10px; border:1px solid #555;'>"
                            f"<span style='color:#AAA; font-weight:bold;'>BẬC 2 - DỰ PHÒNG</span>"
                            f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_backup'][0]:02d} — {res['bac2_backup'][1]:02d}</h2>"
                            f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Bọc Lót Pha 3</p></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("⚡ BẬC 3 VÀ BẬC 4 (ĐẶC THÙ ĂN TIỀN LẺ & HOÀN VỐN)")
            col3, col4 = st.columns(2)
            with col3:
                st.info(f"**BẬC 3 ALPHA:** {' - '.join([f'{n:02d}' for n in res['bac3_alpha']])}\n\n"
                        f"**BẬC 3 BETA:** {' - '.join([f'{n:02d}' for n in res['bac3_beta']])}")
            with col4:
                st.warning(f"**BẬC 4 ALPHA:** {' - '.join([f'{n:02d}' for n in res['bac4_alpha']])}\n\n"
                           f"**BẬC 4 BETA:** {' - '.join([f'{n:02d}' for n in res['bac4_beta']])}")

        # TAB 2: BẬC LỚN
        with tab_large:
            st.subheader("🎯 DÀN CHỦ LỰC BẬC 8 (BẢO HIỂM HOÀN TIỀN TRÚNG 0 & TRÚNG 4/8)")
            b8_str = "  •  ".join([f"**{n:02d}**" for n in res["bac8_master"]])
            st.markdown(
                f"<div style='text-align: center; padding: 20px; background-color: #0D1117; border-radius: 12px; border: 2px solid #7928CA; box-shadow: 0 0 20px rgba(121, 40, 202, 0.4);'>"
                f"<h2 style='color: #FF0080; margin:0;'>{b8_str}</h2>"
                f"<p style='color: #AAA; margin:8px 0 0 0;'>Tối ưu dải thưởng: Trúng 0, 4, 5, 6, 7, 8</p>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.subheader("🔥 DÀN BẬC 7")
                st.info(f"### **{' - '.join([f'{n:02d}' for n in res['bac7_master']])}**")
            with col_l2:
                st.subheader("🌟 DÀN BẬC 9")
                st.success(f"### **{' - '.join([f'{n:02d}' for n in res['bac9_master']])}**")

            st.markdown("---")
            st.subheader("🛡️ DÀN BỌC LÓT BẬC 8 (BACKUP SET)")
            st.warning(f"### **{' - '.join([f'{n:02d}' for n in res['bac8_backup']])}**")

        # TAB 3: PHÂN TÍCH MA TRẬN
        with tab_analytics:
            st.subheader("📊 BẢNG TÍNH ĐIỂM XUNG LỰC HỘI TỰ XUẤT XUẤT TỪ 5 LỚP PIPELINE")
            df_top = pd.DataFrame({
                "Thứ hạng": [f"Top {i+1}" for i in range(20)],
                "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(20)],
                "Cấu trúc Core": ["Alpha Core" if i % 2 == 0 else "Beta Core (Bộc phát)" for i in range(20)],
                "Điểm Tín Hiệu HD-PME": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(20)]
            })
            st.table(df_top)

    else:
        st.warning(f"Cần tối thiểu {min_window} kỳ dữ liệu để khởi chạy siêu cấu trúc HD-PME (Hiện nhận diện được {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 10-30 kỳ Keno vào khung trên để thực thi siêu cấu trúc v20.0.")
