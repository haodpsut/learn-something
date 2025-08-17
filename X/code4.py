import requests
import json
from datetime import datetime
import time
import sqlite3
import os

# ==============================================================================
# --- CẤU HÌNH ---
# ==============================================================================
HELIUS_RPC_ENDPOINT = "https://mainnet.helius-rpc.com/?api-key=f34dab6e-7e5b-4b15-80f5-71f115e2796b"
DATABASE_FILE = "profiles.db"
KNOWN_TEAM_WALLECTS = {
    "771zujoMQDHMGyBWu363HNHr3PPXbSkA7X5kiQBpPNSz": "Ví Gốc / Mẹ",
    "GnE2cQ455ZyvQcP9xfLoVEo4orhyhj8qBz5q1iW1r5nt": "Creator 1 ($BALLZ)",
    "7gbnR6f7EyTyMu6ECf4ndnwREG4TCqtruSvD6gEG4gex": "Creator 2",
    "H9bBXbLgkHrrfnZCvEpGBR4ekWaDeL5LjjkT7SFX7qWK": "Creator 3",
}

SYNC_STATE_KEY = "graph_builder_last_processed_index"
# ==============================================================================
# --- MODULES ---
# ==============================================================================
def connect_db():
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    except sqlite3.Error as e:
        print(f"Lỗi khi kết nối DB: {e}")
        return None, None

def get_single_tx_detail(signature: str):
    """
    Lấy chi tiết của MỘT giao dịch duy nhất, có thử lại.
    """
    retries = 5
    for attempt in range(retries):
        try:
            payload = {"jsonrpc": "2.0", "id": "1", "method": "getTransaction", "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]}
            response = requests.post(HELIUS_RPC_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 429:
                print(f" (Rate limit, chờ 30s...)", end='')
                time.sleep(30)
                continue
            response.raise_for_status()
            result = response.json().get('result')
            return result
        except Exception as e:
            print(f" (Lỗi: {e}, chờ 10s...)", end='')
            time.sleep(10)
    print(f"\n[!!!] Bỏ qua signature {signature[:10]} sau {retries} lần thử thất bại.")
    return None

# ==============================================================================
# --- HÀM CHÍNH ---
# ==============================================================================
def build_graph_one_by_one():
    conn, cursor = connect_db()
    if not conn: return
    
    print("="*70)
    print(" BẮT ĐẦU XÂY DỰNG ĐỒ THỊ (CHẾ ĐỘ BỀN BỈ TUYỆT ĐỐI V1.4) ")
    print("="*70)

    # Khởi tạo
    cursor.execute("INSERT OR IGNORE INTO sync_state (key, value) VALUES (?, ?)", (SYNC_STATE_KEY, -1))
    for address, name in KNOWN_TEAM_WALLECTS.items():
        cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)", (address, name, 0))
    conn.commit()

    cursor.execute("SELECT signature FROM raw_transactions ORDER BY blockTime ASC")
    all_signatures = [row['signature'] for row in cursor.fetchall()]
    total_signatures = len(all_signatures)

    cursor.execute("SELECT value FROM sync_state WHERE key = ?", (SYNC_STATE_KEY,))
    last_processed_index = int(cursor.fetchone()['value'])
    start_index = last_processed_index + 1
    
    if start_index >= total_signatures:
        print("\n✅ TOÀN BỘ LỊCH SỬ ĐÃ ĐƯỢỢC XỬ LÝ!")
        conn.close()
        return

    print(f"Bắt đầu từ index {start_index}...")

    cursor.execute("SELECT address FROM profiles")
    known_wallets = {row['address'] for row in cursor.fetchall()}
    
    # Vòng lặp chính: xử lý từng giao dịch một
    for i in range(start_index, total_signatures):
        signature = all_signatures[i]
        
        progress = ((i + 1) / total_signatures) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Xử lý {i+1}/{total_signatures} ({progress:.4f}%) - sig: {signature[:10]}...", end='')
        
        # Lấy chi tiết
        tx_data = get_single_tx_detail(signature)
        if not tx_data:
            print(" -> Lỗi, bỏ qua.")
            continue

        # Logic lây lan
        tx_timestamp = tx_data.get('blockTime', 0)
        meta, accounts = tx_data.get('meta'), tx_data.get('transaction', {}).get('message', {}).get('accountKeys', [])
        if not meta or not accounts: print(" -> Thiếu dữ liệu, bỏ qua."); continue
        pre_balances, post_balances = meta['preBalances'], meta['postBalances']
        senders = {accounts[idx]['pubkey'] for idx, _ in enumerate(accounts) if post_balances[idx] < pre_balances[idx]}
        if any(s in known_wallets for s in senders):
            receivers = {accounts[idx]['pubkey'] for idx, _ in enumerate(accounts) if post_balances[idx] > pre_balances[idx]}
            for receiver in receivers:
                if receiver and receiver not in known_wallets:
                    cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)",(receiver, f"Satellite_{receiver[:4]}", tx_timestamp))
                    known_wallets.add(receiver)
                    print(f" -> ✨ TÌM THẤY VÍ MỚI: {receiver[:10]}...!")
        
        # Cập nhật tiến trình sau MỖI giao dịch thành công
        cursor.execute("UPDATE sync_state SET value = ? WHERE key = ?", (str(i), SYNC_STATE_KEY))
        conn.commit()
        print(" -> OK.")
        time.sleep(0.2) # Nghỉ 0.2 giây giữa mỗi request

if __name__ == "__main__":
    build_graph_one_by_one()