import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GT_FID(nn.Module):
    def __init__(self, seq_input_dim, node_feat_dim, lstm_hidden_dim=128, gnn_hidden_dim=128, num_classes=2):
        super(GT_FID, self).__init__()
        # LSTM branch
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=lstm_hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=False)
        
        # GNN branch: 2-layer GCN example
        self.gcn1 = GCNConv(node_feat_dim, gnn_hidden_dim)
        self.gcn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        
        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim + gnn_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, seqs, seq_lens, graph_batch):
        # seqs: list of (seq_len_i, seq_input_dim) tensors (un-padded sequences)
        # seq_lens: tensor of sequence lengths
        # graph_batch: PyG Batch object with node features and edge_index
        
        # --- LSTM branch ---
        # Pad sequences
        padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # shape: (batch_size, max_seq_len, seq_input_dim)
        packed_input = nn.utils.rnn.pack_padded_sequence(padded_seqs, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_out, (h_n, c_n) = self.lstm(packed_input)  # h_n shape: (num_layers * num_directions, batch, hidden_size)
        
        h_lstm = h_n[-1]  # Take last layer hidden state for all batch (batch_size, lstm_hidden_dim)
        
        # --- GNN branch ---
        x = graph_batch.x  # Node features (all nodes from all graphs in batch)
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch  # batch vector that assigns each node to graph in batch
        
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)
        
        # Aggregate node features to graph feature by global mean pooling
        h_gnn = global_mean_pool(x, batch)  # shape: (batch_size, gnn_hidden_dim)
        
        # --- Fusion ---
        fused = torch.cat([h_lstm, h_gnn], dim=1)  # shape: (batch_size, lstm_hidden_dim + gnn_hidden_dim)
        
        out = self.classifier(fused)
        
        return out

from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.optim as optim
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_count = 0, 0, 0
    for batch in loader:
        seqs = [data.seq.to(device) for data in batch]
        seq_lens = torch.tensor([data.seq_len for data in batch], device=device)
        
        # PyG batch of graphs, so collate batch of Data objects
        graph_batch = batch.to(device)
        
        labels = torch.tensor([data.y for data in batch], device=device)
        
        optimizer.zero_grad()
        outputs = model(seqs, seq_lens, graph_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            seqs = [data.seq.to(device) for data in batch]
            seq_lens = torch.tensor([data.seq_len for data in batch], device=device)
            graph_batch = batch.to(device)
            labels = torch.tensor([data.y for data in batch], device=device)
            
            outputs = model(seqs, seq_lens, graph_batch)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Device:', device)
    
    data = torch.load('gtfid_graph_data.pt')
    print(f'[INFO] Loaded data with length {len(data)}')
    if len(data) == 3:
        train_data, val_data, test_data = data
    elif len(data) == 2:
        train_data, val_data = data
        test_data = None
    else:
        raise ValueError('Data must have 2 or 3 parts')

    
    train_loader = GeoDataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=64)
    test_loader = GeoDataLoader(test_data, batch_size=64)
    
    seq_input_dim = train_data[0].seq.shape[1]
    node_feat_dim = train_data[0].x.shape[1]
    num_classes = 2
    
    model = GT_FID(seq_input_dim, node_feat_dim, lstm_hidden_dim=128, gnn_hidden_dim=128, num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0.0
    for epoch in range(1, 31):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gtfid_model.pt')
            print('[INFO] Model saved.')
    
    # Test final
    model.load_state_dict(torch.load('best_gtfid_model.pt'))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f'[TEST] Loss={test_loss:.4f}, Acc={test_acc:.4f}')

if __name__ == '__main__':
    main()
