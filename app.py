import numpy as np
from scipy.spatial.distance import pdist, squareform

class SimplePersistentHomologySimulator:
    def __init__(self, n_points=30):
        self.n_points = n_points

    def generate_noisy_circle(self, noise=0.1):
        """Tạo đám mây điểm có cấu trúc hình vòng lặp (Circle/Loop) kèm nhiễu trắng."""
        np.random.seed(42)
        angles = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        x = np.cos(angles) + np.random.normal(0, noise, self.n_points)
        y = np.sin(angles) + np.random.normal(0, noise, self.n_points)
        return np.column_stack((x, y))

    def compute_distance_matrix(self, points):
        """Tính ma trận khoảng cách Euclid giữa các điểm trong đám mây."""
        return squareform(pdist(points))

    def simulate_filtration(self, dist_matrix, thresholds):
        """Mô phỏng quá trình tăng dần bán kính lọc epsilon (Filtration)."""
        print("=== MÔ PHỎNG QUÁ TRÌNH LỌC VIETORIS-RIPS ===")
        n = dist_matrix.shape[0]
        
        for eps in thresholds:
            # Tạo ma trận kề nhị phân dựa trên bán kính epsilon
            adj_matrix = (dist_matrix <= eps).astype(int)
            connected_components = np.linalg.matrix_rank(adj_matrix) # Xấp xỉ đơn giản
            
            # Đếm số thành phần liên thông (H0) và cấu trúc xuất hiện
            active_edges = np.sum(adj_matrix) - n
            print(f"Bán kính epsilon = {eps:.2f} | Cạnh kết nối: {active_edges:3d} | Mức độ liên thông tương đối: {connected_components}")

# --- Khởi chạy mô phỏng ---
if __name__ == "__main__":
    simulator = SimplePersistentHomologySimulator(n_points=20)
    point_cloud = simulator.generate_noisy_circle(noise=0.15)
    dist_mat = simulator.compute_distance_matrix(point_cloud)
    
    # Quét qua các mức bán kính lọc khác nhau
    filtration_thresholds = [0.2, 0.4, 0.7, 1.2]
    simulator.simulate_filtration(dist_mat, filtration_thresholds)
