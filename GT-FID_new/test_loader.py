from dataset import ADFALDDataset
from torch.utils.data import DataLoader

dataset = ADFALDDataset(data_dir='preprocessed_adfa', max_len=100)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    x, y = batch
    print(f"Input shape: {x.shape}")  # [B, 100]
    print(f"Label shape: {y.shape}")  # [B]
    break
