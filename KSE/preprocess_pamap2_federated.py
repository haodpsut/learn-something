import os
import numpy as np
import pandas as pd
from scipy.stats import mode

DATA_DIR = './PAMAP2_Dataset/Protocol/'
SUBJECTS = [f'subject{str(i)}.dat' for i in range(101, 110)]  # sửa theo dataset thực tế

WINDOW_SIZE = 100
WINDOW_STEP = 50
MAX_NAN_COL_RATIO = 0.2  # tỷ lệ NaN tối đa cho cột được giữ lại

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
        try:
            label_mode = mode_result.mode[0]
        except Exception:
            label_mode = mode_result

        X_windows.append(features)
        y_windows.append(label_mode)

    if len(X_windows) == 0:
        print("Warning: no valid windows extracted!")
        return np.array([]), np.array([])

    return np.vstack(X_windows), np.array(y_windows)

def process_file(filepath):
    print(f"Processing {filepath}...")

    # Đọc file
    df = pd.read_csv(filepath, sep=' ', header=None, skiprows=7)  # PAMAP2 đặc thù
    df.columns = [f'feat_{i}' for i in range(df.shape[1] - 2)] + ['label', 'timestamp']

    # Loại bỏ cột timestamp nếu không cần
    df.drop(columns=['timestamp'], inplace=True)

    # Bỏ cột NaN quá nhiều
    nan_ratio = df.isnull().mean()
    cols_to_keep = nan_ratio[nan_ratio < MAX_NAN_COL_RATIO].index
    df = df[cols_to_keep]

    # Nội suy + ffill + bfill
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Bỏ dòng vẫn còn NaN (ít thôi)
    df.dropna(inplace=True)

    # Kiểm tra lại NaN
    if df.isnull().values.any():
        print(f"Warning: NaN still present in {os.path.basename(filepath)} after filling!")

    # Bỏ dòng label = NaN hoặc không hợp lệ
    df = df[df['label'].notna()]

    # Lấy sliding windows
    X, y = sliding_windows(df)

    return X, y

def main():
    all_X = []
    all_y = []

    for sub in SUBJECTS:
        filepath = os.path.join(DATA_DIR, sub)
        if not os.path.isfile(filepath):
            print(f"File {filepath} not found, skipping.")
            continue

        X, y = process_file(filepath)

        if X.size == 0 or y.size == 0:
            print(f"Warning: no valid data extracted from {sub}, skipping.")
            continue

        all_X.append(X)
        all_y.append(y)

    if len(all_X) == 0:
        print("No valid data extracted from any subject!")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    # Kiểm tra NaN tổng thể trước khi lưu
    if np.isnan(X_all).any():
        print("Warning: NaN detected in final features array!")
    if np.isnan(y_all).any():
        print("Warning: NaN detected in final labels array!")

    print(f"Final feature shape: {X_all.shape}, label shape: {y_all.shape}")
    print(f"NaNs in features: {np.isnan(X_all).sum()}, NaNs in labels: {np.isnan(y_all).sum()}")

    np.savez_compressed('data_pamap2_federated.npz', X=X_all, y=y_all)

if __name__ == '__main__':
    main()
