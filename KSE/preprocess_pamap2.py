import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from glob import glob
from sklearn.preprocessing import StandardScaler
import pickle

WINDOW_SIZE = 128
STRIDE = 64
DATA_DIR = './PAMAP2_Dataset/Protocol'  # <-- cập nhật đường dẫn chính xác

SELECTED_COLUMNS = [
    'heart_rate',
    'hand_acc_x', 'hand_acc_y', 'hand_acc_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'
]

COLUMN_MAPPING = {
    'sensor_3': 'hand_acc_x', 'sensor_4': 'hand_acc_y', 'sensor_5': 'hand_acc_z',
    'sensor_6': 'hand_gyro_x', 'sensor_7': 'hand_gyro_y', 'sensor_8': 'hand_gyro_z',
    'sensor_15': 'chest_acc_x', 'sensor_16': 'chest_acc_y', 'sensor_17': 'chest_acc_z',
    'sensor_18': 'chest_gyro_x', 'sensor_19': 'chest_gyro_y', 'sensor_20': 'chest_gyro_z',
    'sensor_27': 'ankle_acc_x', 'sensor_28': 'ankle_acc_y', 'sensor_29': 'ankle_acc_z',
    'sensor_30': 'ankle_gyro_x', 'sensor_31': 'ankle_gyro_y', 'sensor_32': 'ankle_gyro_z'
}

ACTIVITY_LABELS = list({
    1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 18
})

def extract_features(window):
    feats = []
    for i in range(window.shape[1]):
        x = window[:, i]
        x = x - np.mean(x)
        norm = np.sum(np.abs(x)) + 1e-8
        feats.extend([
            np.mean(x), np.std(x), np.max(x), np.min(x), np.median(x),
            entropy(np.abs(x / norm))
        ])
    return np.array(feats)

from scipy.stats import mode

def sliding_windows(df):
    feats, labels = [], []
    for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        # Lấy mode activity_id, xử lý kết quả chắc chắn là array
        mode_result = mode(window['activity_id'], keepdims=True)
        window_label = mode_result.mode[0]  # mode_result.mode luôn là array

        x = window[SELECTED_COLUMNS].values
        feats.append(extract_features(x))
        labels.append(int(window_label))
    return np.array(feats), np.array(labels)

def process_all():
    files = glob(os.path.join(DATA_DIR, '*.dat'))
    all_x, all_y = [], []
    for f in files:
        print(f'Processing {f}')
        df = pd.read_csv(f, sep=' ', header=None)
        df = df.dropna()
        df.columns = ['timestamp', 'activity_id', 'heart_rate'] + [f'sensor_{i}' for i in range(51)]
        df = df.rename(columns=COLUMN_MAPPING)
        df = df[df['activity_id'].isin(ACTIVITY_LABELS)]
        df = df[['activity_id'] + SELECTED_COLUMNS].dropna()

        x, y = sliding_windows(df)

        if x.size == 0 or y.size == 0:
            print(f'Warning: no windows generated from {f}, skipping.')
            continue

        all_x.append(x)
        all_y.append(y)

    if len(all_x) == 0:
        raise RuntimeError("No valid windows generated from any file!")

    X = np.vstack(all_x)
    y = np.concatenate(all_y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f'Final feature shape: {X.shape}, label shape: {y.shape}')

    np.savez('pamap2_fused.npz', X=X, y=y)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    process_all()
