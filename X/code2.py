import requests
import json
from datetime import datetime, timedelta
import time
import sqlite3
import os

# ==============================================================================
# --- CẤU HÌNH ---
# ==============================================================================
HELIUS_API_KEYS = [
    # !!! DÁN 20 API KEY CỦA BẠN VÀO ĐÂY, MỖI KEY TRONG CẶP DẤU NGOẶC KÉP VÀ CÁCH NHAU BẰNG DẤU PHẨY !!!
    # Ví dụ:
    # "abc-123-def-456",
    # "ghi-789-jkl-012",
]
current_api_key_index = 0

def get_api_url():
    return f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEYS[current_api_key_index]}"

def rotate_api_key():
    global current_api_key_index
    previous_key_index = current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(HELIUS_API_KEYS)
    print(f"\n[!] Hạn ngạch API Key #{previous_key_index + 1} có thể đã hết. Chuyển sang Key #{current_api_key_index + 1}...")
    time.sleep(5)
    return current_api_key_index == 0 # Trả về True nếu đã xoay hết 1 vòng

DATABASE_FILE = "profiles.db"
DECODED_CACHE_FILE = "decoded_transactions.json"
KNOWN_TEAM_WALLECTS = { "771zujoMQDHMGyBWu363HNHr3PPXbSkA7X5kiQBpPNSz": "Ví Gốc / Mẹ", "GnE2cQ455ZyvQcP9xfLoVEo4orhyhj8qBz5q1iW1r5nt": "Creator 1 ($BALLZ)", "7gbnR6f7EyTyMu6ECf4ndnwREG4TCqtruSvD6gEG4gex": "Creator 2", "H9bBXbLgkHrrfnZCvEpGBR4ekWaDeL5LjjkT7SFX7qWK": "Creator 3", }
PUMP_FUN_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# ==============================================================================
# --- MODULES ---
# ==============================================================================

def get_transactions_details_resilient(signatures: list):
    if not signatures: return []
    all_tx_details, processed_signatures = [], set()
    if os.path.exists(DECODED_CACHE_FILE):
        print(f"✅ Tìm thấy file cache `{DECODED_CACHE_FILE}`. Đang đọc để tiếp tục...")
        try:
            with open(DECODED_CACHE_FILE, "r") as f: all_tx_details = json.load(f)
            for tx in all_tx_details: processed_signatures.add(tx.get('signature'))
            print(f"  -> Đã đọc {len(all_tx_details)} giao dịch từ cache.")
        except Exception as e: print(f"Lỗi đọc cache: {e}. Bắt đầu lại."); all_tx_details, processed_signatures = [], set()

    signatures_to_process = [s for s in signatures if s not in processed_signatures]
    if not signatures_to_process:
        print("✅ Tất cả các chi tiết giao dịch đã có trong cache.")
        return all_tx_details

    print(f"Còn lại {len(signatures_to_process)} chữ ký cần lấy chi tiết từ API...")
    batch_size = 40
    
    i = 0
    while i < len(signatures_to_process):
        batch = signatures_to_process[i:i + batch_size]
        try:
            print(f"  -> Đang xử lý lô (sử dụng Key #{current_api_key_index + 1})... Tổng: {len(all_tx_details)}", end='\r')
            response = requests.post(get_api_url(), json={"transactions": batch}, timeout=120)

            if response.status_code == 429:
                if rotate_api_key(): 
                    print("[!!!] Đã xoay vòng hết tất cả API key nhưng vẫn bị giới hạn. Dừng lại.")
                    break
                continue

            response.raise_for_status()
            
            batch_details = response.json()
            all_tx_details.extend(batch_details)
            with open(DECODED_CACHE_FILE, "w") as f: json.dump(all_tx_details, f)
            
            i += batch_size
            time.sleep(1.5)
        except requests.exceptions.RequestException as e:
            print(f"\nLỗi mạng hoặc lỗi nghiêm trọng: {e}. Dừng lại.")
            break
    print(f"\nLấy chi tiết hoàn tất. Tổng cộng có {len(all_tx_details)} chi tiết giao dịch.")
    return all_tx_details

def connect_db():
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    except sqlite3.Error as e: print(f"Lỗi khi kết nối DB: {e}"); return None, None

def format_time(unix_ts):
    if not unix_ts or unix_ts in [0, float('inf')]: return "N/A"
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M UTC')

# ==============================================================================
# --- HÀM PHÂN TÍCH CHÍNH ---
# ==============================================================================

def process_and_build_graph():
    conn, cursor = connect_db()
    if not conn: return
    print("="*70); print(" BẮT ĐẦU XÂY DỰNG ĐỒ THỊ MẠNG LƯỚI (ANALYZER V4.8) "); print("="*70)

    cursor.execute("SELECT signature FROM raw_transactions ORDER BY blockTime ASC")
    signatures_to_process = [row['signature'] for row in cursor.fetchall()]
    decoded_transactions = get_transactions_details_resilient(signatures_to_process)
    if not decoded_transactions: conn.close(); return

    print(f"\n🔬 Bắt đầu xây dựng hồ sơ từ {len(decoded_transactions)} giao dịch...")
    for address, name in KNOWN_TEAM_WALLECTS.items():
        cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)", (address, name, 0))
    conn.commit()

    cursor.execute("SELECT address FROM profiles")
    known_wallets_in_db = {row['address'] for row in cursor.fetchall()}
    new_profiles_created = 0

    for i, tx_data in enumerate(decoded_transactions):
        if (i + 1) % 1000 == 0: print(f"  -> Đã xử lý {i+1}/{len(decoded_transactions)} giao dịch. Đã tạo {new_profiles_created} hồ sơ mới.", end='\r')
        tx_timestamp = tx_data.get('timestamp', 0)
        
        for transfer in tx_data.get("events", {}).get("nativeTransfers", []):
            sender, receiver = transfer.get("fromUserAccount"), transfer.get("toUserAccount")
            amount_sol = transfer['amount'] / 1e9
            if sender in known_wallets_in_db and receiver and receiver not in known_wallets_in_db:
                cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)", (receiver, f"Satellite_{receiver[:4]}", tx_timestamp))
                known_wallets_in_db.add(receiver); new_profiles_created += 1
            if receiver:
                cursor.execute("UPDATE profiles SET sol_funded = sol_funded + ?, last_activity_unix = max(ifnull(last_activity_unix, 0), ?) WHERE address = ?", (amount_sol, tx_timestamp, receiver))

        fee_payer = tx_data.get("feePayer")
        if fee_payer in known_wallets_in_db:
            cursor.execute("UPDATE profiles SET total_tx = total_tx + 1, last_activity_unix = max(ifnull(last_activity_unix, 0), ?) WHERE address = ?", (tx_timestamp, fee_payer))
            if tx_data.get("type") == "SWAP": cursor.execute("UPDATE profiles SET swap_count = swap_count + 1 WHERE address = ?", (fee_payer,))
            if "pump.fun" in tx_data.get("source", "").lower():
                for inst in tx_data.get("transaction", {}).get("message", {}).get("instructions", []):
                    if inst.get("programId") == PUMP_FUN_PROGRAM_ID and len(inst.get("accounts", [])) > 6:
                        creator, token_mint = inst["accounts"][6], inst["accounts"][0]
                        if creator == fee_payer:
                            cursor.execute("UPDATE profiles SET pump_fun_creations = pump_fun_creations + 1 WHERE address = ?", (creator,))
                            cursor.execute("INSERT OR IGNORE INTO created_tokens (mint_address, creator_address, creation_date_unix, status) VALUES (?, ?, ?, ?)", (token_mint, creator, tx_timestamp, 'New'))
                            break
        if (i + 1) % 500 == 0: conn.commit()
        
    conn.commit()
    print(f"\n✅ Xây dựng/cập nhật hồ sơ hoàn tất. Tổng cộng đã tạo {new_profiles_created} hồ sơ mới.")
    
    print("\n🔄 Đang cập nhật trạng thái và vai trò tổng hợp...")
    cursor.execute("SELECT address, total_tx, pump_fun_creations, swap_count, last_activity_unix FROM profiles")
    all_profiles_data = cursor.fetchall()
    seven_days_ago_unix = (datetime.now() - timedelta(days=7)).timestamp()
    updates_to_perform = []
    for profile in all_profiles_data:
        roles = set()
        if profile['last_activity_unix'] and profile['last_activity_unix'] < seven_days_ago_unix: roles.add("Sleeper")
        else: roles.add("Active")
        if profile['pump_fun_creations'] > 0: roles.add("Creator")
        if profile['swap_count'] > 5: roles.add("Trader")
        updates_to_perform.append((json.dumps(list(roles)), profile['address']))
    
    cursor.executemany("UPDATE profiles SET roles = ? WHERE address = ?", updates_to_perform)
    conn.commit()
    print(f"✅ Đã cập nhật vai trò cho {len(all_profiles_data)} hồ sơ.")
    
    conn.close()
    
    print("\n" + "="*70); print(" BÁO CÁO NHANH SAU KHI XỬ LÝ "); print("="*70)
    conn, cursor = connect_db()
    if conn:
        cursor.execute("SELECT COUNT(*) FROM profiles"); total_profiles = cursor.fetchone()[0]
        print(f"📊 TỔNG SỐ HỒ SƠ TRONG DATABASE: {total_profiles}")
        cursor.execute("SELECT COUNT(*), roles FROM profiles GROUP BY roles")
        print("\n📊 PHÂN BỐ VAI TRÒ HIỆN TẠI:")
        for row in cursor.fetchall(): print(f"  - {row['roles'] if row['roles'] else 'Chưa xác định'}: {row['COUNT(*)']} ví")
        conn.close()

    print("\n" + "="*70); print(" HOÀN TẤT PHÂN TÍCH. "); print("="*70)

if __name__ == "__main__":
    if not HELIUS_API_KEYS or "key_1..." in HELIUS_API_KEYS[0]:
        print("="*70); print("!!! LỖI CẤU HÌNH !!!"); print("Vui lòng mở file script và điền các API Key thật vào danh sách `HELIUS_API_KEYS`."); print("="*70)
    else:
        process_and_build_graph()