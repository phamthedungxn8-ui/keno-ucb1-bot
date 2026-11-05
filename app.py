import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.fft import fft

class AnonymousSpatialEngine:
    def __init__(self, n_positions=80, n_drawn=20):
        self.D = n_positions
        self.K = n_drawn

    def vector_to_positions(self, binary_vector: np.ndarray) -> np.ndarray:
        """Trích xuất các chỉ số vị trí (1-80) mang giá trị 1."""
        return np.where(binary_vector == 1)[0] + 1

    def compute_wasserstein_series(self, matrix: np.ndarray) -> list:
        """
        1. TÍNH KHOẢNG CÁCH WASSERSTEIN (W1) GIỮA CÁC KỲ LIÊN TIẾP
        Đo lường 'công' tối thiểu để biến đổi hình thái tập hợp kỳ t-1 sang kỳ t.
        """
        T, D = matrix.shape
        w1_scores = []
        for t in range(1, T):
            pos_prev = self.vector_to_positions(matrix[t-1])
            pos_curr = self.vector_to_positions(matrix[t])
            # Tính khoảng cách Wasserstein giữa 2 phân bố vị trí
            dist = wasserstein_distance(pos_prev, pos_curr)
            w1_scores.append(dist)
        return w1_scores

    def compute_von_neumann_entropy(self, matrix: np.ndarray) -> np.ndarray:
        """
        2. TÍNH ENTROPY VƯỚNG VÍU VON NEUMANN
        Chia 80 vị trí thành 2 phân vùng A (1-40) và B (41-80) để đo độ lệch nén năng lượng.
        """
        T, D = matrix.shape
        half = D // 2
        entropies = []

        for t in range(T):
            vec = matrix[t]
            prob_A = vec[:half].sum() / self.K
            prob_B = vec[half:].sum() / self.K

            # Tạo ma trận mật độ rút gọn 2x2 cho phân vùng
            probs = np.array([prob_A, prob_B])
            probs = probs[probs > 0] # Lọc các giá trị bằng 0 để tránh lỗi log
            
            # Tính Von Neumann Entropy: S = - sum(p * log2(p))
            S = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
            entropies.append(S)

        return np.array(entropies)

    def compute_spatial_fourier_spectrum(self, matrix: np.ndarray) -> np.ndarray:
        """
        3. THUẬT TOÁN BIẾN ĐỔI FOURIER KHÔNG GIANG (DFT)
        Quét các sóng hài tần số vị trí trong lồng quay.
        """
        T, D = matrix.shape
        # Tính biến đổi Fourier trên từng kỳ và lấy trung bình biên độ công suất
        fft_amplitudes = np.abs(fft(matrix, axis=1))
        power_spectrum = np.mean(fft_amplitudes**2, axis=0)
        return power_spectrum[:D // 2] # Trả về nửa phổ dương (Do tính đối xứng)

    def analyze_swarm_signals(self, matrix: np.ndarray):
        """
        ĐIỀU PHỐI VÀ TỔNG HỢP TÍN HIỆU TỪ 3 TÁC TỬ
        """
        T, D = matrix.shape
        w1_series = self.compute_wasserstein_series(matrix)
        entropy_series = self.compute_von_neumann_entropy(matrix)
        fourier_spectrum = self.compute_spatial_fourier_spectrum(matrix)

        # 1. Tín hiệu Wasserstein: Phát hiện thời điểm nén/giãn dải
        latest_w1 = w1_series[-1] if len(w1_series) > 0 else 0
        w1_trend = "COMPRESSING (Co cụm dải)" if latest_w1 < np.mean(w1_series) else "EXPANDING (Bùng nổ dải)"

        # 2. Tín hiệu Entropy: Xác định xem năng lượng có bị lệch phân vùng
        latest_entropy = entropy_series[-1]
        entropy_status = "BALANCED (Entropy cực đại)" if latest_entropy > 0.95 else "LOCALIZED (Nén lệch pha)"

        # 3. Tín hiệu Fourier: Tìm tần số sóng hài thống trị (Dominant Spatial Frequency)
        # Bỏ qua k=0 (thành phần DC)
        dominant_k = np.argmax(fourier_spectrum[1:]) + 1
        harmonic_period = round(D / dominant_k, 2)

        return {
            "w1_series": w1_series,
            "latest_w1": round(latest_w1, 4),
            "w1_trend": w1_trend,
            "latest_entropy": round(latest_entropy, 4),
            "entropy_status": entropy_status,
            "dominant_spatial_k": dominant_k,
            "harmonic_period_positions": harmonic_period,
            "fourier_spectrum": fourier_spectrum
        }

# ==============================================================================
# KHUNG KIỂM ĐỊNH GIẢ LẬP VỚI DỮ LIỆU TẬP HỢP
# ==============================================================================
if __name__ == "__main__":
    # Giả lập 5 kỳ quay (mỗi kỳ chọn 20 quả bóng ngẫu nhiên từ 80 vị trí)
    np.random.seed(42)
    T_test = 5
    sample_matrix = np.zeros((T_test, 80))
    
    for t in range(T_test):
        drawn_indices = np.random.choice(80, 20, replace=False)
        sample_matrix[t, drawn_indices] = 1.0

    # Khởi chạy cỗ máy phân tích không gian
    engine = AnonymousSpatialEngine(n_positions=80, n_drawn=20)
    results = engine.analyze_swarm_signals(sample_matrix)

    print("==========================================================")
    print("🌐 KẾT QUẢ PHÂN TÍCH TÔ-PÔ VÀ KHÔNG GIANG TRẠNG THÁI (v115.0)")
    print("==========================================================")
    print(f"📊 Khoảng cách Wasserstein (W1) kỳ mới nhất : {results['latest_w1']} -> [{results['w1_trend']}]")
    print(f"🌀 Entropy Von Neumann (Phân vùng A/B)       : {results['latest_entropy']} -> [{results['entropy_status']}]")
    print(f"📻 Tần số sóng hài không gian ưu thế (k)      : Bậc k = {results['dominant_spatial_k']}")
    print(f"📏 Chu kỳ khoảng cách vị trí rơi ưu thế       : Cách nhau ~{results['harmonic_period_positions']} vị trí")
    print("==========================================================")
