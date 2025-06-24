#!/usr/bin/env python3
"""
build_graph_hetero.py
-------------------
Tạo đồ thị đa hình (HeteroData) từ:
  • Email (node type 'email')
  • Domain (node type 'domain')
  • URL   (node type 'url')
  • IP    (node type 'ip')

Edges:
  email -> domain: sent_by (From), to (To/Cc/Bcc)
  email -> url: contains_url
  email -> ip: contains_ip

Email-node features:
  • Content embedding (Sentence-BERT)
  • Ratio uppercase chars
  • Ratio digit chars
  • Subject length

Domain-node features:
  • Embedding of domain string (SBERT)

URL/IP-node features: zero vector

Ghi ra: graph_data_hetero.pt
"""

import os, glob, argparse, email, re
import numpy as np, torch, networkx as nx
from tqdm import tqdm
from email.utils import getaddresses, parseaddr
from email.policy import compat32
from email.parser import BytesParser
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

# Parser for robust header parsing
PARSER = BytesParser(policy=compat32)
# Regex for URLs and IPv4
URL_REGEX = re.compile(r"https?://[\w./?=&%-]+", re.IGNORECASE)
IP_REGEX  = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")

# Utils

def domain_of(addr: str):
    addr = parseaddr(addr)[1].lower()
    return addr.split("@")[1] if "@" in addr else None

# Email-level recipients extraction

def recipients(msg):
    hdrs = []
    for key in ("to","cc","bcc","resent-to"):
        hdrs += msg.get_all(key, [])
    return [addr for _, addr in getaddresses(hdrs)]

# Text payload

def text_body(msg, limit=4000):
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/plain":
                return (p.get_payload(decode=True) or b"")[:limit]
    return (msg.get_payload(decode=True) or b"")[:limit]

# Safe mbox iterator

def safe_mbox(path):
    with open(path, 'rb') as fh:
        buf = []
        for line in fh:
            if line.startswith(b"From ") and buf:
                yield PARSER.parsebytes(b"".join(buf))
                buf = []
            buf.append(line)
        if buf:
            yield PARSER.parsebytes(b"".join(buf))

# Safe maildir iterator

def safe_maildir(root):
    for f in glob.iglob(os.path.join(root, '**/*'), recursive=True):
        if os.path.isfile(f):
            try:
                with open(f, 'rb') as fh:
                    yield PARSER.parse(fh)
            except:
                continue

# Main build

def build_hetero(naz_root, enron_root, out_path):
    # Load SBERT
    sbert = SentenceTransformer(
        'all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

    emails = []
    edges_email_domain = []
    edges_email_recips = []
    edges_email_url = []
    edges_email_ip = []
    domain_set = set()
    url_set = set()
    ip_set = set()

    email_idx = 0
    # Nazario (label=1)
    for mbox in glob.iglob(os.path.join(naz_root, '*.mbox')):
        for msg in tqdm(safe_mbox(mbox), desc=f"Nazario {os.path.basename(mbox)}"):
            subj_raw = msg.get('Subject', '') or ''
            subj = str(subj_raw)
            body_bytes = text_body(msg)
            body = body_bytes.decode('latin1', errors='ignore')
            emb = sbert.encode(body, normalize_embeddings=True)
            up_ratio  = sum(1 for c in body if c.isupper())/max(1,len(body))
            dig_ratio = sum(1 for c in body if c.isdigit())/max(1,len(body))
            sub_len   = len(subj)
            feat = torch.tensor(np.concatenate([emb, [up_ratio, dig_ratio, sub_len]]), dtype=torch.float)
            emails.append((feat, 1))
            frm_field = msg.get('From','')
            frm_str   = str(frm_field) if isinstance(frm_field, email.headerregistry.Header) else frm_field
            frm_dom = domain_of(frm_str)
            if frm_dom:
                edges_email_domain.append((email_idx, frm_dom)); domain_set.add(frm_dom)
            for addr in recipients(msg):
                dom = domain_of(addr)
                if dom:
                    edges_email_recips.append((email_idx, dom)); domain_set.add(dom)
            for url in URL_REGEX.findall(body):
                edges_email_url.append((email_idx, url)); url_set.add(url)
            for ip in IP_REGEX.findall(body):
                edges_email_ip.append((email_idx, ip)); ip_set.add(ip)
            email_idx += 1
    # Enron (label=0)
    for msg in tqdm(safe_maildir(enron_root), desc="Enron"):
        subj_raw = msg.get('Subject', '') or ''
        subj = str(subj_raw)
        body_bytes = text_body(msg)
        body = body_bytes.decode('latin1', errors='ignore')
        emb = sbert.encode(body, normalize_embeddings=True)
        up_ratio  = sum(1 for c in body if c.isupper())/max(1,len(body))
        dig_ratio = sum(1 for c in body if c.isdigit())/max(1,len(body))
        sub_len   = len(subj)
        feat = torch.tensor(np.concatenate([emb, [up_ratio, dig_ratio, sub_len]]), dtype=torch.float)
        emails.append((feat, 0))
        frm_field = msg.get('From','')
        frm_str   = str(frm_field) if isinstance(frm_field, email.headerregistry.Header) else frm_field
        frm_dom = domain_of(frm_str)
        if frm_dom:
            edges_email_domain.append((email_idx, frm_dom)); domain_set.add(frm_dom)
        for addr in recipients(msg):
            dom = domain_of(addr)
            if dom:
                edges_email_recips.append((email_idx, dom)); domain_set.add(dom)
        for url in URL_REGEX.findall(body):
            edges_email_url.append((email_idx, url)); url_set.add(url)
        for ip in IP_REGEX.findall(body):
            edges_email_ip.append((email_idx, ip)); ip_set.add(ip)
        email_idx += 1

    # index nodes
    domain_list = list(domain_set)
    url_list    = list(url_set)
    ip_list     = list(ip_set)
    dom_map = {d:i for i,d in enumerate(domain_list)}
    url_map = {u:i for i,u in enumerate(url_list)}
    ip_map  = {i:i for i in ip_list}

    data = HeteroData()
    feats = torch.stack([feat for feat,_ in emails])
    labs  = torch.tensor([lab for _,lab in emails], dtype=torch.long)
    data['email'].x = feats; data['email'].y = labs
    dom_emb = torch.tensor(sbert.encode(domain_list, normalize_embeddings=True), dtype=torch.float)
    data['domain'].x = dom_emb
    data['url'].x    = torch.zeros((len(url_list), feats.size(1)), dtype=torch.float)
    data['ip'].x     = torch.zeros((len(ip_list), feats.size(1)), dtype=torch.float)

    # edges
    data['email','sent_by','domain'].edge_index = torch.tensor(
      [[e for e,d in edges_email_domain], [dom_map[d] for e,d in edges_email_domain]],
      dtype=torch.long)
    data['email','to','domain'].edge_index = torch.tensor(
      [[e for e,d in edges_email_recips],[dom_map[d] for e,d in edges_email_recips]],
      dtype=torch.long)
    data['email','contains_url','url'].edge_index = torch.tensor(
      [[e for e,u in edges_email_url],[url_map[u] for e,u in edges_email_url]],
      dtype=torch.long)
    data['email','contains_ip','ip'].edge_index = torch.tensor(
      [[e for e,i in edges_email_ip],[ip_map[i] for e,i in edges_email_ip]],
      dtype=torch.long)

    torch.save(data, out_path)
    print(f"✓ Saved hetero graph to {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nazario_dir', required=True)
    ap.add_argument('--enron_root',   required=True)
    ap.add_argument('--out',          default='graph_data_hetero.pt')
    args = ap.parse_args()
    build_hetero(args.nazario_dir, args.enron_root, args.out)
