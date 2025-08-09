import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Bắt đầu quá trình chuẩn bị dữ liệu từ các file local...")

# --- BƯỚC MỚI: ĐỌC VÀ GỘP DỮ LIỆU TỪ NHIỀU FILE ---
data_dir = 'cicids2017_data' # Thư mục chứa các file CSV
# Tìm tất cả các file có đuôi .csv trong thư mục
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

if not all_files:
    print(f"Lỗi: Không tìm thấy file CSV nào trong thư mục '{data_dir}'.")
    print("Hãy chạy script download_kaggle_data.py trước.")
    exit()

print(f"Tìm thấy {len(all_files)} file CSV. Bắt đầu đọc và gộp...")

# Đọc từng file và gộp chúng vào một DataFrame duy nhất
li = []
for filename in all_files:
    # Thêm encoding='latin1' để tránh lỗi khi đọc một số file
    df_file = pd.read_csv(filename, index_col=None, header=0, encoding='latin1')
    li.append(df_file)

df = pd.concat(li, axis=0, ignore_index=True)
print("Gộp dữ liệu thành công!")

# --- CÁC BƯỚC TIẾP THEO GIỮ NGUYÊN NHƯ TRƯỚC ---

# 1.2. Làm sạch dữ liệu cơ bản
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 1.3. Tạo bộ dữ liệu con (Subset) cho bài toán nhị phân
print(f"\nSố lượng dòng dữ liệu tổng cộng sau khi gộp và làm sạch: {len(df)}")
print("Phân bố các nhãn:\n", df['Label'].value_counts().head()) # In 5 nhãn phổ biến nhất

df_subset = df[df['Label'].isin(['BENIGN', 'DDoS'])].copy()

# 1.4. Lấy mẫu cân bằng (Balanced Sampling)
n_samples_per_class = 2000 # Có thể tăng số mẫu lên một chút vì dữ liệu nhiều hơn
random_state = 42

df_benign = df_subset[df_subset['Label'] == 'BENIGN'].sample(n=n_samples_per_class, random_state=random_state)
df_ddos = df_subset[df_subset['Label'] == 'DDoS'].sample(n=n_samples_per_class, random_state=random_state)

df_balanced = pd.concat([df_benign, df_ddos])

print(f"\nĐã tạo bộ dữ liệu con cân bằng với {len(df_balanced)} mẫu.")
print("Phân bố nhãn trong bộ dữ liệu mới:\n", df_balanced['Label'].value_counts())

# 1.5. Tách thành đặc trưng (X) và nhãn (y)
X = df_balanced.drop('Label', axis=1)
y = df_balanced['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# 1.6. Phân chia dữ liệu thành tập Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state, stratify=y # Tăng test_size lên 0.3
)
print(f"\nKích thước tập huấn luyện (train): {X_train.shape}")
print(f"Kích thước tập kiểm thử (test): {X_test.shape}")

# --- BƯỚC 2: TIỀN XỬ LÝ VÀ GIẢM CHIỀU ---

# 2.1. Chuẩn hóa dữ liệu (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nĐã chuẩn hóa dữ liệu.")

# 2.2. Giảm chiều dữ liệu với PCA
n_components = 4
pca = PCA(n_components=n_components, random_state=random_state)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Đã giảm chiều dữ liệu xuống còn {n_components} thành phần chính.")
print(f"Kích thước X_train sau PCA: {X_train_pca.shape}")
print(f"Kích thước X_test sau PCA: {X_test_pca.shape}")

explained_variance = pca.explained_variance_ratio_.sum()
print(f"Tổng phương sai được giải thích bởi {n_components} thành phần: {explained_variance:.2%}")

print("\n--- HOÀN TẤT CHUẨN BỊ DỮ LIỆU ---")
print("Các biến đã sẵn sàng cho mô hình Machine Learning:")
print("X_train_pca, y_train, X_test_pca, y_test")


# Thêm vào cuối file process_data.py

# --- BƯỚC MỚI: LƯU DỮ LIỆU ĐÃ XỬ LÝ RA FILE ---
output_dir = 'processed_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lưu các mảng numpy
np.save(os.path.join(output_dir, 'X_train.npy'), X_train_pca)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values) # Lưu y_train dưới dạng numpy array
np.save(os.path.join(output_dir, 'X_test.npy'), X_test_pca)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values) # Lưu y_test dưới dạng numpy array

print(f"\nĐã lưu dữ liệu đã xử lý vào thư mục '{output_dir}'.")
