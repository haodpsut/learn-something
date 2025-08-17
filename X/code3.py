import requests
import json
from datetime import datetime
import time
import sqlite3

# ==============================================================================
# --- CẤU HÌNH ---
# ==============================================================================
# Chỉ cần 1 API Key cho việc này, vì RPC call rất rẻ
HELIUS_RPC_ENDPOINT = "https://mainnet.helius-rpc.com/?api-key=f34dab6e-7e5b-4b15-80f5-71f115e2796b"
DATABASE_FILE = "profiles.db"
KNOWN_TEAM_WALLECTS = {
    "771zujoMQDHMGyBWu363HNHr3PPXbSkA7X5kiQBpPNSz": "Ví Gốc / Mẹ",
    "GnE2cQ455ZyvQcP9xfLoVEo4orhyhj8qBz5q1iW1r5nt": "Creator 1 ($BALLZ)",
    "7gbnR6f7EyTyMu6ECf4ndnwREG4TCqtruSvD6gEG4gex": "Creator 2",
    "H9bBXbLgkHrrfnZCvEpGBR4ekWaDeL5LjjkT7SFX7qWK": "Creator 3",
}
# --- CẤU HÌNH MỚI ---
PROCESSING_CHUNK_SIZE = 1000 # Xử lý 1000 giao dịch mỗi lần chạy
SYNC_STATE_KEY_FOR_GRAPH_BUILDER = "graph_builder_last_processed_index"

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

def get_simple_txs_batch(signatures: list):
    """Lấy chi tiết giao dịch cơ bản bằng RPC call, trả về danh sách đã giải mã."""
    all_details = []
    batch_size = 50 # Có thể tăng batch size cho RPC call
    total_batches = -(-len(signatures) // batch_size)
    
    for i in range(0, len(signatures), batch_size):
        batch = signatures[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        print(f"  -> Đang lấy chi tiết RPC lô {current_batch_num}/{total_batches}...", end='\r')
        try:
            payload = [{"jsonrpc": "2.0", "id": sig, "method": "getTransaction", "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]} for sig in batch]
            response = requests.post(HELIUS_RPC_ENDPOINT, json=payload, timeout=90)
            response.raise_for_status()
            
            results = response.json()
            for res in results:
                if res and res.get('result'):
                    all_details.append(res['result'])
            time.sleep(0.5) # Độ trễ nhỏ
        except Exception as e:
            print(f"\nLỗi khi lấy chi tiết RPC lô {current_batch_num}: {e}")
    print("\nLấy chi tiết RPC hoàn tất.")
    return all_details

# ==============================================================================
# --- HÀM CHÍNH ---
# ==============================================================================
def build_relationship_graph():
    conn, cursor = connect_db()
    if not conn: return
    
    print("="*70)
    print(" BẮT ĐẦU XÂY DỰNG KHUNG XƯƠNG ĐỒ THỊ (GRAPH BUILDER V1.0) ")
    print("="*70)

    # Lấy toàn bộ chữ ký từ DB
    cursor.execute("SELECT signature FROM raw_transactions ORDER BY blockTime ASC")
    all_signatures = [row['signature'] for row in cursor.fetchall()]
    
    if not all_signatures:
        print("Không có giao dịch thô nào trong DB. Vui lòng chạy scraper.py trước."); conn.close(); return

    # Khởi tạo hồ sơ cốt lõi
    for address, name in KNOWN_TEAM_WALLECTS.items():
        cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)", (address, name, 0))
    conn.commit()

    # Lấy chi tiết cơ bản bằng RPC
    all_tx_details = get_simple_txs_batch(all_signatures)
    
    print(f"\n🔬 Bắt đầu xây dựng hồ sơ từ {len(all_tx_details)} giao dịch...")
    cursor.execute("SELECT address FROM profiles")
    known_wallets = {row['address'] for row in cursor.fetchall()}
    new_profiles_created = 0

    for i, tx_data in enumerate(all_tx_details):
        if (i + 1) % 1000 == 0:
            print(f"  -> Đã xử lý {i+1}/{len(all_tx_details)} giao dịch. Đã tạo {new_profiles_created} hồ sơ mới.", end='\r')
        
        tx_timestamp = tx_data.get('blockTime', 0)
        meta = tx_data.get('meta')
        accounts = tx_data.get('transaction', {}).get('message', {}).get('accountKeys', [])
        
        if not meta or not accounts: continue

        pre_balances = meta['preBalances']
        post_balances = meta['postBalances']
        
        senders = {accounts[i]['pubkey'] for i, _ in enumerate(accounts) if post_balances[i] < pre_balances[i]}
        
        # Logic lây lan
        if any(s in known_wallets for s in senders):
            receivers = {accounts[i]['pubkey'] for i, _ in enumerate(accounts) if post_balances[i] > pre_balances[i]}
            for receiver in receivers:
                if receiver and receiver not in known_wallets:
                    cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)",
                                   (receiver, f"Satellite_{receiver[:4]}", tx_timestamp))
                    known_wallets.add(receiver)
                    new_profiles_created += 1

    conn.commit()
    print(f"\n✅ Xây dựng đồ thị hoàn tất. Tổng cộng đã tạo {new_profiles_created} hồ sơ mới.")
    
    cursor.execute("SELECT COUNT(*) FROM profiles")
    print(f"📊 TỔNG SỐ HỒ SƠ TRONG DATABASE HIỆN TẠI: {cursor.fetchone()[0]}")
    
    conn.close()

if __name__ == "__main__":
    build_relationship_graph()