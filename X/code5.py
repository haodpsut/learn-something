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
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
API_REQUESTS_PER_SECOND = 4 # Tốc độ an toàn để tránh rate limit
MIN_FUNDING_THRESHOLD_SOL = 0.05 # Ngưỡng tối thiểu để coi là một giao dịch cấp vốn

# ==============================================================================
# --- MODULES ---
# ==============================================================================

def connect_db():
    """Tạo kết nối đến database và trả về connection, cursor."""
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    except sqlite3.Error as e:
        print(f"Lỗi khi kết nối DB: {e}")
        return None, None

def is_personal_wallet(wallet_address: str):
    """Kiểm tra xem một địa chỉ có phải là ví cá nhân (sở hữu bởi System Program)."""
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo", "params": [wallet_address, {"encoding": "jsonParsed"}]}
        response = requests.post(HELIUS_RPC_ENDPOINT, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json().get('result', {}).get('value')
        if result and result.get('owner') == SYSTEM_PROGRAM_ID:
            return True
    except Exception:
        return False
    return False

def get_single_tx_detail(signature: str):
    """Lấy chi tiết của MỘT giao dịch duy nhất, có cơ chế thử lại."""
    retries = 5
    for attempt in range(retries):
        try:
            payload = {"jsonrpc": "2.0", "id": "1", "method": "getTransaction", "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]}
            response = requests.post(HELIUS_RPC_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 429:
                wait_time = 20 * (attempt + 1)
                print(f" (Rate limit, chờ {wait_time}s...)", end='')
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json().get('result')
        except Exception as e:
            print(f" (Lỗi: {e}, chờ 10s...)", end='')
            time.sleep(10)
    print(f"\n[!!!] Bỏ qua signature {signature[:10]} sau {retries} lần thử thất bại.")
    return None

def format_time(unix_ts):
    """Chuyển đổi unix timestamp sang string định dạng YYYY-MM-DD HH:MM UTC."""
    if not unix_ts or unix_ts in [0, float('inf')]: return "N/A"
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M UTC')
    
# ==============================================================================
# --- HÀM CHÍNH ---
# ==============================================================================
def build_graph_with_gatekeeper():
    conn, cursor = connect_db()
    if not conn: return
    
    print("="*70)
    print(" BẮT ĐẦU XÂY DỰNG ĐỒ THỊ (LOGIC GATEKEEPER V1.8 - FINAL) ")
    print("="*70)

    # Khởi tạo sync_state nếu chưa có
    cursor.execute("INSERT OR IGNORE INTO sync_state (key, value) VALUES (?, ?)", (SYNC_STATE_KEY, -1))
    conn.commit()

    cursor.execute("SELECT signature FROM raw_transactions ORDER BY blockTime ASC")
    all_signatures = [row['signature'] for row in cursor.fetchall()]
    total_signatures = len(all_signatures)

    cursor.execute("SELECT value FROM sync_state WHERE key = ?", (SYNC_STATE_KEY,))
    last_processed_index = int(cursor.fetchone()['value'])
    start_index = last_processed_index + 1
    
    if start_index >= total_signatures:
        print("\n✅ TOÀN BỘ LỊCH SỬ ĐÃ ĐƯỢC XỬ LÝ!")
        conn.close()
        return

    print(f"Tiếp tục từ index {start_index}...")

    # Tải toàn bộ danh sách ví đã biết vào bộ nhớ để tăng tốc
    cursor.execute("SELECT address FROM profiles")
    known_wallets = {row['address'] for row in cursor.fetchall()}
    
    for i in range(start_index, total_signatures):
        signature = all_signatures[i]
        
        progress = ((i + 1) / total_signatures) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Xử lý {i+1}/{total_signatures} ({progress:.4f}%) - sig: {signature[:10]}...", end='')
        
        tx_data = get_single_tx_detail(signature)
        if not tx_data:
            print(" -> Lỗi, bỏ qua.")
            continue

        tx_timestamp = tx_data.get('blockTime', 0)
        instructions = tx_data.get('transaction', {}).get('message', {}).get('instructions', [])
        new_wallets_found_this_tx = 0
        
        for inst in instructions:
            if inst.get('program') == 'system' and inst.get('parsed', {}).get('type') == 'transfer':
                info = inst['parsed']['info']
                sender, receiver = info.get('source'), info.get('destination')
                amount_sol = info.get('lamports', 0) / 1e9
                
                # Áp dụng 3 lớp bảo vệ
                if (sender in known_wallets and receiver and 
                    receiver not in known_wallets and 
                    amount_sol >= MIN_FUNDING_THRESHOLD_SOL):
                    
                    # Lớp bảo vệ cuối cùng: Kiểm tra "chính chủ"
                    if is_personal_wallet(receiver):
                        cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)",
                                       (receiver, f"Satellite_{receiver[:4]}", tx_timestamp))
                        known_wallets.add(receiver)
                        new_wallets_found_this_tx += 1
                    else:
                        print(f" -> ⚠️  Bỏ qua Program Account: {receiver[:10]}...", end='')

        # Cập nhật tiến trình sau mỗi giao dịch
        cursor.execute("UPDATE sync_state SET value = ? WHERE key = ?", (str(i), SYNC_STATE_KEY))
        conn.commit()

        if new_wallets_found_this_tx > 0:
            print(f" -> ✨ TÌM THẤY {new_wallets_found_this_tx} VÍ MỚI!")
        else:
            print(" -> OK.")
            
        time.sleep(1.0 / API_REQUESTS_PER_SECOND)

if __name__ == "__main__":
    build_graph_with_gatekeeper()