import numpy as np

def check_fl_data(npz_path):
    data = np.load(npz_path)
    X = data['X']
    y = data['y']  # Đây là 1D array nhãn

    print(f"Loaded X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Loaded y shape: {y.shape}, dtype: {y.dtype}")

    # Nếu y là 1D thì không thể tách client id từ y
    # Bạn cần client_ids riêng, giả sử bạn có lưu client_ids trong npz hoặc file khác

    # Nếu không có client_ids, ta chỉ kiểm tra nhãn
    labels = y.astype(int)

    print(f"Total samples: {len(labels)}")

    unique_labels = np.unique(labels)
    print(f"Total unique classes (labels) in dataset: {len(unique_labels)}")
    print(f"Class labels range from {unique_labels.min()} to {unique_labels.max()}")

    print(f"NaNs in X: {np.isnan(X).sum()}")
    print(f"NaNs in labels: {np.isnan(labels).sum()}")

    # Nếu không có client_ids, không thể kiểm tra phân chia theo client
    # Nếu có client_ids, bạn load thêm rồi xử lý phân chia như trước

    # In phân bố lớp trên toàn bộ dataset
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution sample (class: count):")
    for cls, cnt in zip(unique[:10], counts[:10]):
        print(f"  {cls}: {cnt}")

if __name__ == "__main__":
    npz_file = "data_pamap2_federated.npz"
    check_fl_data(npz_file)
