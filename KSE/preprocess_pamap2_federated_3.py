import os
import numpy as np
import pandas as pd
from scipy.stats import mode

DATA_DIR = './PAMAP2_Dataset/Protocol/'
SUBJECTS = [f'subject{i}.dat' for i in range(101, 110)]  # subjects 101 - 109

WINDOW_SIZE = 100
WINDOW_STEP = 50
MAX_NAN_COL_RATIO = 0.2  # max ratio NaN để giữ cột

from scipy.stats import mode

def sliding_windows(df):
    X_windows = []
    y_windows = []

    num_rows = df.shape[0]
    for start in range(0, num_rows - WINDOW_SIZE + 1, WINDOW_STEP):
        window = df.iloc[start:start+WINDOW_SIZE]

        if window.isnull().values.any():
            print(f"Warning: NaN detected in window features at index {start}, skipping this window.")
            continue

        features = window.drop(columns=['label']).values.flatten()

        mode_result = mode(window['label'])
        # Xử lý mode_result có thể là scalar hoặc có .mode
        try:
            label_mode = mode_result.mode[0]
        except (AttributeError, IndexError):
            # Nếu mode_result là scalar
            label_mode = mode_result

        X_windows.append(features)
        y_windows.append(label_mode)

    if len(X_windows) == 0:
        print("Warning: no valid windows extracted!")
        return np.array([]), np.array([])

    return np.vstack(X_windows), np.array(y_windows)

def process_file(filepath):
    print(f"Processing {filepath}...")

    # Đọc dữ liệu, PAMAP2 có header 7 dòng cần bỏ
    df = pd.read_csv(filepath, sep=' ', header=None, skiprows=7)

    # Đặt tên cột: tất cả cột trừ 2 cuối cùng là feature, cuối là label và timestamp
    df.columns = [f'feat_{i}' for i in range(df.shape[1]-2)] + ['label', 'timestamp']

    # Bỏ cột timestamp
    df.drop(columns=['timestamp'], inplace=True)

    # Loại bỏ các cột có NaN quá nhiều (> MAX_NAN_COL_RATIO)
    nan_ratio = df.isnull().mean()
    cols_to_keep = nan_ratio[nan_ratio < MAX_NAN_COL_RATIO].index
    df = df[cols_to_keep]

    # Nội suy linear + forward fill + backward fill để lấp NaN
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Bỏ các dòng còn NaN
    df.dropna(inplace=True)

    # Bỏ dòng label bị NaN (nếu còn)
    df = df[df['label'].notna()]

    # Lấy sliding windows features và labels
    X, y = sliding_windows(df)

    return X, y

def main():
    all_X = []
    all_y = []
    all_client_ids = []

    for sub in SUBJECTS:
        filepath = os.path.join(DATA_DIR, sub)
        if not os.path.isfile(filepath):
            print(f"File {filepath} not found, skipping.")
            continue

        X, y = process_file(filepath)
        if X.size == 0 or y.size == 0:
            print(f"Warning: no valid data extracted from {sub}, skipping.")
            continue

        # Gán client id theo subject number, ví dụ 'subject101.dat' → client_id = 101
        client_id = int(sub.replace('subject', '').replace('.dat', ''))

        client_ids = np.full(len(y), client_id, dtype=int)

        all_X.append(X)
        all_y.append(y)
        all_client_ids.append(client_ids)

    if len(all_X) == 0:
        print("No valid data extracted from any subject!")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    client_ids_all = np.concatenate(all_client_ids)

    print(f"Final X shape: {X_all.shape}")
    print(f"Final y shape: {y_all.shape}")
    print(f"Final client_ids shape: {client_ids_all.shape}")

    # Kiểm tra NaN
    if np.isnan(X_all).any():
        print("Warning: NaN detected in features!")
    if np.isnan(y_all).any():
        print("Warning: NaN detected in labels!")
    if np.isnan(client_ids_all).any():
        print("Warning: NaN detected in client IDs!")

    # Kiểm tra client ID unique
    unique_clients = np.unique(client_ids_all)
    print(f"Number of unique clients: {len(unique_clients)}")
    print(f"Unique client IDs: {unique_clients}")

    # Gộp label + client_id thành ma trận 2 cột (float cho label, int cho client_id)
    y_combined = np.column_stack((y_all, client_ids_all))

    print(f"y_combined shape: {y_combined.shape}, dtype: {y_combined.dtype}")

    # Lưu file npz
    np.savez_compressed('data_pamap2_federated.npz', X=X_all, y=y_combined)

if __name__ == '__main__':
    main()
