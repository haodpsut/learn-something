import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Fix seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Constants
MAX_LEN = 150
PAD_ID = 0
VOCAB_SIZE = 400
BATCH_SIZE = 64
EPOCHS = 30
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 2
MAX_GRAD_NORM = 2.0

# Dataset class unchanged
class SystemCallDataset(Dataset):
    def __init__(self, data_list, label_list, max_len=MAX_LEN):
        self.samples = []
        for seq in data_list:
            if len(seq) < max_len:
                seq = seq + [PAD_ID] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            self.samples.append(seq)
        self.labels = label_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx]), torch.LongTensor([self.labels[idx]])

def load_sequences_from_folder(folder_path, label):
    all_data, all_labels = [], []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    seq = list(map(int, f.read().strip().split()))
                    all_data.append(seq)
                    all_labels.append(label)
    return all_data, all_labels

def build_dataset():
    normal_path = './Training_Data_Master'
    attack_path = './Attack_Data_Master'
    normal_seqs, normal_labels = load_sequences_from_folder(normal_path, 0)
    attack_seqs, attack_labels = load_sequences_from_folder(attack_path, 1)
    all_seqs = normal_seqs + attack_seqs
    all_labels = normal_labels + attack_labels
    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)
    return list(all_seqs), list(all_labels)

def split_dataset(dataset):
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

# Models unchanged
class GRUBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0)), h.squeeze(0)  # Return hidden for t-SNE

class TransformerBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x_pooled = x.mean(dim=1)
        return self.fc(x_pooled), x_pooled  # Return pooled embedding for t-SNE

def plot_tsne(embeddings, labels, epoch, save_path):
    tsne = TSNE(n_components=2, random_state=SEED)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    labels = np.array(labels)
    for label in np.unique(labels):
        idxs = labels == label
        plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=f'Class {label}', alpha=0.6, s=10)
    plt.legend()
    plt.title(f't-SNE Visualization Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def train(model, train_loader, val_loader, device, epochs, log_path, tsne_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    model.to(device)

    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)

    with open(log_path, 'w') as log_file:
        log_file.write("epoch,train_loss,train_acc,val_loss,val_acc,precision,recall,f1_score\n")

        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for seq, label in train_loader:
                seq, label = seq.to(device), label.to(device).squeeze()
                optimizer.zero_grad()
                out, _ = model(seq)
                loss = criterion(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_loss += loss.item() * label.size(0)
                preds = out.argmax(dim=1)
                total_correct += (preds == label).sum().item()
                total_samples += label.size(0)

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            val_loss, val_acc, val_preds, val_labels, val_embeddings = evaluate(model, val_loader, device, criterion, return_preds=True)

            precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Precision: {precision:.4f} "
                  f"| Recall: {recall:.4f} | F1: {f1:.4f}")

            log_file.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")
            log_file.flush()

            # Save t-SNE figure
            plot_tsne(val_embeddings, val_labels, epoch, os.path.join(tsne_dir, f"tsne_epoch_{epoch}.png"))

            scheduler.step(val_acc)

def evaluate(model, loader, device, criterion=None, return_preds=False):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for seq, label in loader:
            seq, label = seq.to(device), label.to(device).squeeze()
            out, embedding = model(seq)
            if criterion is not None:
                loss = criterion(out, label)
                total_loss += loss.item() * label.size(0)
            preds = out.argmax(dim=1)
            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_embeddings.append(embedding.cpu().numpy())

    avg_loss = total_loss / total_samples if criterion is not None else 0
    accuracy = total_correct / total_samples
    if return_preds:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), all_embeddings
    else:
        return avg_loss, accuracy

def run_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    sequences, labels = build_dataset()
    dataset = SystemCallDataset(sequences, labels)
    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # GRU
    print("\n=== GRU Baseline ===")
    model_gru = GRUBaseline()
    train(model_gru, train_loader, val_loader, device, EPOCHS, 'gru_training_log.csv', 'figures/gru')
    test_loss, test_acc, _, _, _ = evaluate(model_gru, test_loader, device, criterion=nn.CrossEntropyLoss(), return_preds=True)
    print(f"[TEST] GRU Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    # Transformer
    print("\n=== Transformer Baseline ===")
    model_trans = TransformerBaseline()
    train(model_trans, train_loader, val_loader, device, EPOCHS, 'transformer_training_log.csv', 'figures/transformer')
    test_loss, test_acc, _, _, _ = evaluate(model_trans, test_loader, device, criterion=nn.CrossEntropyLoss(), return_preds=True)
    print(f"[TEST] Transformer Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    run_baselines()
