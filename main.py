import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from numba import njit
import math

class KenoQuantumEngine:
    def __init__(self, draw_count=500):
        self.draw_count = draw_count

    def fetch_live_data(self):
        """
        Cào dữ liệu Keno thực tế từ Minh Ngọc (Trả về Ma trận Nhị phân + Mã kỳ quay mới nhất)
        """
        url = "https://www.minhngoc.com.vn/ket-qua-xo-so/vietlott/keno.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            tables = soup.find_all('table', class_='bkqkeno')
            draws = []
            latest_draw_id = "N/A"
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    draw_id_elem = row.find('td', class_='str_kieu') or row.find('a')
                    if draw_id_elem and latest_draw_id == "N/A":
                        latest_draw_id = draw_id_elem.text.strip()

                    numbers_td = row.find_all('td', class_='day_so')
                    if numbers_td:
                        nums = [int(td.text.strip()) for td in numbers_td if td.text.strip().isdigit()]
                        if len(nums) == 20:
                            draws.append(nums)
            
            if len(draws) > 0:
                matrix = self._to_binary_matrix(draws[:self.draw_count])
                draw_id = latest_draw_id if latest_draw_id != "N/A" else "#LATEST_LIVE"
                return matrix, draw_id
            else:
                matrix = self._generate_mock_data(self.draw_count)
                return matrix, "#MOCK_DATA"
                
        except Exception:
            matrix = self._generate_mock_data(self.draw_count)
            return matrix, "#MOCK_DATA"

    def _to_binary_matrix(self, draws):
        matrix = np.zeros((len(draws), 80), dtype=np.int32)
        for i, draw in enumerate(draws):
            for num in draw:
                if 1 <= num <= 80:
                    matrix[i, num - 1] = 1
        return matrix

    def _generate_mock_data(self, n_draws):
        np.random.seed(None)
        matrix = np.zeros((n_draws, 80), dtype=np.int32)
        for i in range(n_draws):
            cols = np.random.choice(80, 20, replace=False)
            matrix[i, cols] = 1
        return matrix

@njit
def compute_cooccurrence_b2(matrix):
    n_draws, n_nums = matrix.shape
    co_matrix = np.zeros((n_nums, n_nums), dtype=np.float64)
    for i in range(n_nums):
        for j in range(i + 1, n_nums):
            count = 0
            for k in range(n_draws):
                if matrix[k, i] == 1 and matrix[k, j] == 1:
                    count += 1
            co_matrix[i, j] = count / n_draws
            co_matrix[j, i] = co_matrix[i, j]
    return co_matrix

def calculate_entropy(prob):
    if prob <= 0 or prob >= 1:
        return 0.0
    return - (prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))

def optimize_bac_2_3(matrix):
    n_draws = matrix.shape[0]
    freq = np.sum(matrix, axis=0) / n_draws
    co_matrix = compute_cooccurrence_b2(matrix)

    results_b2 = []
    for i in range(80):
        for j in range(i + 1, 80):
            p_joint = co_matrix[i, j]
            p_indep = freq[i] * freq[j]
            if p_indep > 0:
                lift = p_joint / p_indep
                h_val = calculate_entropy(p_joint)
                ev = (p_joint * 6.0) - 1.0
                results_b2.append((i + 1, j + 1, p_joint, lift, h_val, ev))

    df_b2 = pd.DataFrame(results_b2, columns=['Num1', 'Num2', 'Prob', 'Lift', 'Entropy', 'EV'])
    df_b2 = df_b2.sort_values(by=['EV', 'Lift'], ascending=False).reset_index(drop=True)

    top_pairs = df_b2.head(30)[['Num1', 'Num2']].values
    results_b3 = []
    
    for pair in top_pairs:
        n1, n2 = int(pair[0] - 1), int(pair[1] - 1)
        for n3 in range(80):
            if n3 == n1 or n3 == n2:
                continue
            p_b3 = np.sum((matrix[:, n1] == 1) & (matrix[:, n2] == 1) & (matrix[:, n3] == 1)) / n_draws
            p_b2_match = np.sum(((matrix[:, n1] == 1) & (matrix[:, n2] == 1)) | 
                                ((matrix[:, n1] == 1) & (matrix[:, n3] == 1)) | 
                                ((matrix[:, n2] == 1) & (matrix[:, n3] == 1))) / n_draws
            
            ev_b3 = (p_b3 * 40.0) + (p_b2_match * 4.0) - 1.0
            results_b3.append((n1 + 1, n2 + 1, n3 + 1, p_b3, ev_b3))

    df_b3 = pd.DataFrame(results_b3, columns=['Num1', 'Num2', 'Num3', 'Prob_3_3', 'EV'])
    df_b3['Tuple'] = df_b3.apply(lambda r: tuple(sorted([int(r['Num1']), int(r['Num2']), int(r['Num3'])])), axis=1)
    df_b3 = df_b3.drop_duplicates(subset=['Tuple']).sort_values(by='EV', ascending=False).reset_index(drop=True)

    return df_b2.head(10), df_b3.head(10)
