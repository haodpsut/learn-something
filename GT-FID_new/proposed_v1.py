import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")

def collate_batch(batch):
    graphs = Batch.from_data_list(batch)
    seqs = torch.nn.utils.rnn.pad_sequence([data.seq for data in batch], batch_first=True, padding_value=0)
    seq_lens = torch.tensor([data.seq_len.item() for data in batch], dtype=torch.long)
    labels = torch.tensor([data.y.item() for data in batch], dtype=torch.long)
    return seqs.to(device), seq_lens.to(device), graphs.to(device), labels.to(device)



class GT_FID(nn.Module):
    def __init__(self, seq_vocab_size, seq_emb_dim=128, lstm_hidden=256, gcn_hidden=128, fused_dim=384, num_classes=2, dropout=0.3):
        super().__init__()
        self.seq_emb = nn.Embedding(seq_vocab_size, seq_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(seq_emb_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.gcn1 = GCNConv(1, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.bn_gcn = nn.BatchNorm1d(gcn_hidden)
        self.fc_fuse = nn.Linear(lstm_hidden*2 + gcn_hidden, fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, seqs, seq_lens, graph_batch):
        emb = self.seq_emb(seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        h_lstm = torch.cat([hn[-2], hn[-1]], dim=1)  # bidirectional

        x = graph_batch.x.float()
        edge_index = graph_batch.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.bn_gcn(x)
        x = F.relu(self.gcn2(x, edge_index))

        batch_index = graph_batch.batch
        h_gcn = torch.zeros((graph_batch.num_graphs, x.size(1)), device=x.device)
        for i in range(graph_batch.num_graphs):
            mask = (batch_index == i)
            if mask.sum() > 0:
                h_gcn[i] = x[mask].mean(dim=0)

        fused = torch.cat([h_lstm, h_gcn], dim=1)
        fused = self.dropout(self.fc_fuse(fused))
        out = self.classifier(fused)
        return out



def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for seqs, seq_lens, graphs, labels in loader:
        optimizer.zero_grad()
        outputs = model(seqs, seq_lens, graphs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, seq_lens, graphs, labels in loader:
            outputs = model(seqs, seq_lens, graphs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def main():
    data = torch.load('gtfid_graph_data.pt')
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    max_syscall_id = 0
    for d in train_data + val_data + test_data:
        max_syscall_id = max(max_syscall_id, d.seq.max().item())
    seq_vocab_size = max_syscall_id + 1

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_batch)

    model = GT_FID(seq_vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    for epoch in range(1, 31):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)
        end = time.time()

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {end - start:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gtfid_model.pth')

    # Test với best model
    model.load_state_dict(torch.load('best_gtfid_model.pth'))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
    print(f"[TEST] Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
