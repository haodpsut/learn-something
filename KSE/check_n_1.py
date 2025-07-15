# check_fl_data.py
import numpy as np
from collections import Counter

def check_data(path):
    data = np.load(path)
    X = data['X']
    y = data['y']

    labels = y[:, 0].astype(int)
    clients = y[:, 1].astype(int)

    print(f"Total samples: {len(labels)}")
    print(f"Unique classes: {np.unique(labels)}")
    print(f"Unique clients: {np.unique(clients)}")

    for cid in np.unique(clients):
        mask = clients == cid
        client_labels = labels[mask]
        label_dist = Counter(client_labels)
        print(f"\nClient {cid} - {mask.sum()} samples - {len(label_dist)} classes")
        for lbl, cnt in sorted(label_dist.items()):
            print(f"  Label {lbl}: {cnt} samples")

check_data("data_pamap2_federated.npz")
