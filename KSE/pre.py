import os
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

DATA_DIR = './PAMAP2_Dataset/Protocol/'
SUBJECTS = [f'subject{str(i)}.dat' for i in range(101, 110)]

WINDOW_SIZE = 100
WINDOW_STEP = 50
MAX_NAN_COL_RATIO = 0.2

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
        mode_val = mode_result.mode
        if isinstance(mode_val, np.ndarray):
            label_mode = mode_val[0]
        else:
            label_mode = mode_val

        X_windows.append(features)
        y_windows.append(label_mode)

    if len(X_windows) == 0:
        print("Warning: no valid windows extracted!")
        return np.array([]), np.array([])

    return np.vstack(X_windows), np.array(y_windows)


def process_file(filepath):
    print(f"Processing {filepath}...")

    df = pd.read_csv(filepath, sep=' ', header=None, skiprows=7)
    df.columns = [f'feat_{i}' for i in range(df.shape[1] - 2)] + ['label', 'timestamp']

    df.drop(columns=['timestamp'], inplace=True)

    nan_ratio = df.isnull().mean()
    cols_to_keep = nan_ratio[nan_ratio < MAX_NAN_COL_RATIO].index
    df = df[cols_to_keep]

    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.dropna(inplace=True)

    if df.isnull().values.any():
        print(f"Warning: NaN still present in {os.path.basename(filepath)} after filling!")

    df = df[df['label'].notna()]

    X, y = sliding_windows(df)
    return X, y

def main():
    all_X = []
    all_y_raw = []
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

        client_id = int(sub.replace('subject', '').replace('.dat', ''))
        client_ids = np.full(len(y), client_id, dtype=int)

        all_X.append(X)
        all_y_raw.append(y)
        all_client_ids.append(client_ids)

    if len(all_X) == 0:
        print("No valid data extracted from any subject!")
        return

    X_all = np.vstack(all_X)
    y_all_raw = np.concatenate(all_y_raw)  # 1D array of raw labels
    client_ids_all = np.concatenate(all_client_ids)

    print(f"Raw labels sample: {y_all_raw[:20]}")
    print(f"Unique raw labels count: {len(np.unique(y_all_raw))}")

    # Encode labels to integers (1D)
    le = LabelEncoder()
    y_all_encoded = le.fit_transform(y_all_raw)

    # Kiểm tra NaN
    if np.isnan(X_all).any():
        print("Warning: NaN detected in final features array!")
    if np.isnan(y_all_encoded).any():
        print("Warning: NaN detected in final labels array!")

    print(f"Final feature shape: {X_all.shape}, label shape: {y_all_encoded.shape}")
    print(f"NaNs in features: {np.isnan(X_all).sum()}, NaNs in labels: {np.isnan(y_all_encoded).sum()}")

    unique_clients = np.unique(client_ids_all)
    print(f"Number of clients (subjects): {len(unique_clients)}")
    print(f"Unique client IDs: {unique_clients}")

    # Lưu dữ liệu theo chuẩn: X (features), y (labels), client_ids (IDs)
    np.savez_compressed(
        'data_pamap2_federated.npz',
        X=X_all,
        y=y_all_encoded,
        client_ids=client_ids_all
    )
    print("Saved data_pamap2_federated.npz successfully.")

if __name__ == '__main__':
    main()
