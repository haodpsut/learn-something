import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder
import argparse
import csv

# Increase CSV field size limit
csv.field_size_limit(10**6)

# Argument parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='TREC-05.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_embeddings', type=str, default='embeddings.npy',
                        help='Output path for saved embeddings (npy memmap)')
    parser.add_argument('--output_labels', type=str, default='labels.pt',
                        help='Output path for saved labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding')
    parser.add_argument('--max_length', type=int, default=256, help='Max token length')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top K senders/receivers to one-hot; others as "<OTH>"')
    args = parser.parse_args()
    return args

# Feature Processor with top-K limiting
def compute_handcrafted(df, top_k):
    # Limit cardinality for sender and receiver
    df['sender'] = df['sender'].fillna('')
    df['receiver'] = df['receiver'].fillna('')
    top_s = df['sender'].value_counts().nlargest(top_k).index.tolist()
    top_r = df['receiver'].value_counts().nlargest(top_k).index.tolist()
    df['sender2'] = df['sender'].where(df['sender'].isin(top_s), '<OTH>')
    df['receiver2'] = df['receiver'].where(df['receiver'].isin(top_r), '<OTH>')

    # One-hot both
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    sr_df = df[['sender2','receiver2']]
    enc.fit(sr_df)
    sr_feat = enc.transform(sr_df)

    # URL binary
    url_feat = (df['urls'].astype(str).str.strip() != '').astype(int).to_numpy().reshape(-1,1)

    # Date normalized
    dates = pd.to_datetime(df['date'], errors='coerce', utc=True)
    dates = dates.fillna(pd.Timestamp('1970-01-01', tz='UTC'))
    ts = (dates.astype('int64') // 10**9).to_numpy().reshape(-1,1)
    ts = (ts - ts.mean()) / (ts.std() + 1e-6)

    # Combine
    feat = np.hstack([sr_feat, url_feat, ts])
    return feat, enc

# Main
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and filter data
    print("Loading CSV...")
    df = pd.read_csv(args.csv_path, engine='python', on_bad_lines='skip')
    df = df.dropna(subset=['body','label'])
    df = df[df['label'].isin([0,1,'0','1'])].copy()
    df['label'] = df['label'].astype(int)
    df.columns = [c.lower() for c in df.columns]
    n = len(df)

    # Compute handcrafted features
    print(f"Computing handcrafted features with top_k={args.top_k}...")
    handcrafted, enc = compute_handcrafted(df, args.top_k)
    hc_dim = handcrafted.shape[1]

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    model.eval()

    # Determine emb_dim with dummy
    with torch.no_grad():
        sample = tokenizer([df['body'].iloc[0]], return_tensors='pt',
                            truncation=True, padding=True, max_length=args.max_length)
        sample = {k:v.to(device) for k,v in sample.items()}
        emb_dim = model(**sample).last_hidden_state.size(2)

    # Create memmap
    total_dim = emb_dim + hc_dim
    print(f"Allocating memmap of shape {(n, total_dim)}...")
    emb_memmap = np.memmap(args.output_embeddings, dtype='float32', mode='w+', shape=(n, total_dim))

    # Extract and combine
    print("Extracting embeddings and writing memmap...")
    for i in tqdm(range(0, n, args.batch_size)):
        batch_texts = df['body'].iloc[i:i+args.batch_size].astype(str).tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           return_tensors='pt', max_length=args.max_length)
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            txt_emb = model(**inputs).last_hidden_state[:,0,:].cpu().numpy()
        emb_memmap[i:i+txt_emb.shape[0], :emb_dim] = txt_emb
        emb_memmap[i:i+txt_emb.shape[0], emb_dim:] = handcrafted[i:i+txt_emb.shape[0]]

    emb_memmap.flush()
    print(f"Saved embeddings to {args.output_embeddings}")

    # Save labels
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    torch.save(labels, args.output_labels)
    print(f"Saved labels to {args.output_labels}")

if __name__ == '__main__':
    main()
