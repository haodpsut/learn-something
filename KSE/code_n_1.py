import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from scipy.stats import mode

WINDOW_SIZE = 100
STEP_SIZE = 50
VALID_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]
DATA_DIR = './PAMAP2_Dataset/Protocol/'

def sliding_windows(df):
    X = []
    y = []
    data = df.values
    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[i:i+WINDOW_SIZE, :]
        label_window = df['activityID'].values[i:i+WINDOW_SIZE]
        label_mode = mode(label_window, keepdims=True).mode[0]
        X.append(window[:, 3:])  # IMU data only
        y.append(label_mode)
    return np.array(X), np.array(y)

def process_file(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.replace(to_replace='nan', value=np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.columns = ['timestamp', 'activityID', 'heart_rate'] + [f'col{i}' for i in range(4, 55)]
    df = df[df['activityID'].isin(VALID_ACTIVITIES)]
    df['subject'] = int(file_path.split('subject')[-1].split('.')[0])
    X, y = sliding_windows(df)
    clients = np.full((len(y),), df['subject'].iloc[0])
    return X, y, clients

def main():
    X_all = []
    y_all = []
    client_ids = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith('.dat'):
            print(f"Processing {os.path.join(DATA_DIR, fname)}...")
            X, y, clients = process_file(os.path.join(DATA_DIR, fname))
            X_all.append(X)
            y_all.append(y)
            client_ids.append(clients)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    client_ids = np.concatenate(client_ids)

    print("Raw labels sample:", y_all[:20])
    print("Unique raw labels count:", len(np.unique(y_all)))

    le = LabelEncoder()
    y_all_encoded = le.fit_transform(y_all)
    print("After encoding: unique labels:", np.unique(y_all_encoded))

    # Mapping từ activity ID gốc → label mới
    print("== Mapping activityID -> encoded label ==")
    for original, encoded in zip(le.classes_, le.transform(le.classes_)):
        print(f"ActivityID {original} => Encoded label {encoded}")

    # Gộp label với clientID để lưu
    y_combined = np.vstack([y_all_encoded, client_ids]).T

    print("Final feature shape:", X_all.shape, "label shape:", y_combined.shape)
    print("NaNs in features:", np.isnan(X_all).sum(), ", NaNs in labels:", np.isnan(y_combined).sum())
    print("Number of clients (subjects):", len(np.unique(client_ids)))
    print("Unique client IDs:", np.unique(client_ids))

    np.savez("data_pamap2_federated.npz", X=X_all, y=y_combined)

    print("Saved data_pamap2_federated.npz successfully.")

if __name__ == '__main__':
    main()
