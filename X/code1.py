import requests
import json
from datetime import datetime, timedelta
import time
import sqlite3
import os

# ==============================================================================
# --- CẤU HÌNH ---
# ==============================================================================
HELIUS_API_KEY = "b7136879-e7d5-47f7-9aff-d2640767f758"
HELIUS_API_URL = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"
DATABASE_FILE = "profiles.db"
DECODED_CACHE_FILE = "decoded_transactions.json"

KNOWN_TEAM_WALLECTS = {
    "771zujoMQDHMGyBWu363HNHr3PPXbSkA7X5kiQBpPNSz": "Ví Gốc / Mẹ",
    "GnE2cQ455ZyvQcP9xfLoVEo4orhyhj8qBz5q1iW1r5nt": "Creator 1 ($BALLZ)",
    "7gbnR6f7EyTyMu6ECf4ndnwREG4TCqtruSvD6gEG4gex": "Creator 2",
    "H9bBXbLgkHrrfnZCvEpGBR4ekWaDeL5LjjkT7SFX7qWK": "Creator 3",
}
PUMP_FUN_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# ==============================================================================
# --- MODULES ---
# ==============================================================================



# === HÀM get_transactions_details ĐÃ ĐƯỢC NÂNG CẤP ===
def get_transactions_details_robust(signatures: list):
    if not signatures: return []
    all_tx_details, processed_signatures = [], set()
    if os.path.exists(DECODED_CACHE_FILE):
        print(f"✅ Tìm thấy file cache `{DECODED_CACHE_FILE}`. Đang đọc để tiếp tục...")
        try:
            with open(DECODED_CACHE_FILE, "r") as f: all_tx_details = json.load(f)
            for tx in all_tx_details: processed_signatures.add(tx['signature'])
            print(f"  -> Đã đọc {len(all_tx_details)} giao dịch từ cache.")
        except Exception: all_tx_details, processed_signatures = [], set()

    signatures_to_process = [s for s in signatures if s not in processed_signatures]
    if not signatures_to_process:
        print("✅ Tất cả các chi tiết giao dịch đã có trong cache.")
        return all_tx_details

    print(f"Còn lại {len(signatures_to_process)} chữ ký cần lấy chi tiết từ API...")
    batch_size = 40; total_batches = -(-len(signatures_to_process) // batch_size)
    
    for i in range(0, len(signatures_to_process), batch_size):
        batch = signatures_to_process[i:i + batch_size]; current_batch_num = (len(processed_signatures) // batch_size) + (i // batch_size) + 1
        retries = 10
        for attempt in range(retries):
            print(f"  -> Đang xử lý lô {current_batch_num} (thử lần {attempt + 1})...", end='\r')
            try:
                response = requests.post(HELIUS_API_URL, json={"transactions": batch}, timeout=120)
                if response.status_code == 429: raise requests.exceptions.HTTPError(f"429 Too Many Requests")
                response.raise_for_status()
                batch_details = response.json(); all_tx_details.extend(batch_details)
                with open(DECODED_CACHE_FILE, "w") as f: json.dump(all_tx_details, f)
                print(f"  -> Xử lý lô {current_batch_num}/{total_batches + (len(processed_signatures) // batch_size)} thành công. Đã lưu cache. Tổng: {len(all_tx_details)} giao dịch.   ")
                time.sleep(2)
                break
            except requests.exceptions.RequestException as e:
                wait_time = 60
                print(f"\nLỗi tại lô {current_batch_num}: {e}. Chờ {wait_time} giây để thử lại...")
                if attempt < retries - 1: time.sleep(wait_time)
                else: print(f"\n[!!!] Bỏ qua lô {current_batch_num} sau {retries} lần thử thất bại.")
    print(f"\nLấy chi tiết hoàn tất. Tổng cộng có {len(all_tx_details)} chi tiết giao dịch.")
    return all_tx_details


def get_transactions_details(signatures: list):
    """
    Lấy chi tiết giao dịch với cơ chế thử lại thông minh để xử lý rate limit.
    """
    if not signatures: return []
    batch_size = 50
    all_tx_details = []
    total_batches = -(-len(signatures) // batch_size)
    print(f"Bắt đầu lấy chi tiết cho {len(signatures)} chữ ký...")

    for i in range(0, len(signatures), batch_size):
        batch = signatures[i:i + batch_size]
        current_batch_num = i//batch_size + 1
        
        # Logic thử lại
        retries = 5
        for attempt in range(retries):
            print(f"  -> Đang xử lý lô {current_batch_num}/{total_batches} (thử lần {attempt + 1})...", end='\r')
            try:
                response = requests.post(HELIUS_API_URL, json={"transactions": batch}, timeout=90)
                
                # Nếu gặp lỗi 429, ném ra một exception để kích hoạt cơ chế thử lại
                if response.status_code == 429:
                    raise requests.exceptions.HTTPError(f"429 Too Many Requests")

                response.raise_for_status() # Ném lỗi cho các status code 4xx/5xx khác
                
                all_tx_details.extend(response.json())
                time.sleep(1.2) # Tăng nhẹ độ trễ để "lịch sự" hơn
                break # Nếu thành công, thoát khỏi vòng lặp thử lại
            
            except requests.exceptions.RequestException as e:
                print(f"\nLỗi tại lô {current_batch_num}: {e}. Đang chờ để thử lại...")
                if attempt < retries - 1:
                    time.sleep(15) # Chờ 15 giây trước khi thử lại
                else:
                    print(f"\n[!!!] Bỏ qua lô {current_batch_num} sau {retries} lần thử thất bại.")

    print(f"\nLấy chi tiết hoàn tất. Lấy được {len(all_tx_details)}/{len(signatures)} chi tiết giao dịch.")
    return all_tx_details


def connect_db():
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    except sqlite3.Error as e:
        print(f"Lỗi khi kết nối DB: {e}")
        return None, None

def format_time(unix_ts):
    if not unix_ts or unix_ts in [0, float('inf')]: return "N/A"
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M UTC')

# ==============================================================================
# --- HÀM PHÂN TÍCH CHÍNH ---
# ==============================================================================

def process_and_build_graph():
    conn, cursor = connect_db()
    if not conn: return

    print("="*70)
    print(" BẮT ĐẦU XÂY DỰNG ĐỒ THỊ MẠNG LƯỚI (ANALYZER V4.7) ")
    print("="*70)

    # --- Bước 1: Lấy chi tiết giao dịch (ưu tiên cache) ---
    decoded_transactions = []
    if os.path.exists(DECODED_CACHE_FILE):
        print(f"✅ Tìm thấy file cache `{DECODED_CACHE_FILE}`. Đang đọc dữ liệu...")
        with open(DECODED_CACHE_FILE, "r") as f:
            decoded_transactions = json.load(f)
    else:
        print("Không tìm thấy file cache. Sẽ lấy dữ liệu từ API (quá trình này sẽ lâu)...")
        cursor.execute("SELECT signature FROM raw_transactions ORDER BY blockTime ASC")
        signatures_to_process = [row['signature'] for row in cursor.fetchall()]
        if signatures_to_process:
            decoded_transactions = get_transactions_details_robust(signatures_to_process)
            print(f"Đang lưu {len(decoded_transactions)} giao dịch vào cache...")
            with open(DECODED_CACHE_FILE, "w") as f: json.dump(decoded_transactions, f)
            print("✅ Đã lưu vào cache thành công.")

    if not decoded_transactions:
        print("Không có dữ liệu chi tiết để xử lý."); conn.close(); return

    # --- Bước 2: Xây dựng đồ thị (Graph Building) ---
    print(f"\n🔬 Bắt đầu xây dựng hồ sơ từ {len(decoded_transactions)} giao dịch...")
    
    # Khởi tạo hồ sơ cho các ví cốt lõi
    for address, name in KNOWN_TEAM_WALLECTS.items():
        cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)", (address, name, 0))
    conn.commit()

    # Bắt đầu thuật toán "lây lan"
    known_wallets_in_db = set(KNOWN_TEAM_WALLECTS.keys())
    new_profiles_created = 0

    for i, tx_data in enumerate(decoded_transactions):
        if (i + 1) % 1000 == 0:
            print(f"  -> Đã xử lý {i+1}/{len(decoded_transactions)} giao dịch. Đã tạo {new_profiles_created} hồ sơ mới.", end='\r')
        
        tx_timestamp = tx_data.get('timestamp', 0)
        
        # 1. Logic lây lan: Tìm ví mới được cấp vốn
        for transfer in tx_data.get("events", {}).get("nativeTransfers", []):
            sender, receiver = transfer.get("fromUserAccount"), transfer.get("toUserAccount")
            amount_sol = transfer['amount'] / 1e9
            
            # Nếu người gửi đã có trong hồ sơ và người nhận chưa có -> thêm người nhận vào
            if sender in known_wallets_in_db and receiver and receiver not in known_wallets_in_db:
                cursor.execute("INSERT OR IGNORE INTO profiles (address, name, first_contact_unix) VALUES (?, ?, ?)",
                               (receiver, f"Satellite_{receiver[:4]}", tx_timestamp))
                known_wallets_in_db.add(receiver)
                new_profiles_created += 1

            # Cập nhật thông tin cho người nhận (dù mới hay cũ)
            if receiver:
                cursor.execute("UPDATE profiles SET sol_funded = sol_funded + ?, last_activity_unix = max(ifnull(last_activity_unix, 0), ?) WHERE address = ?", 
                               (amount_sol, tx_timestamp, receiver))

        # 2. Cập nhật các bộ đếm hành vi cho người trả phí
        fee_payer = tx_data.get("feePayer")
        if fee_payer and fee_payer in known_wallets_in_db:
            cursor.execute("UPDATE profiles SET total_tx = total_tx + 1, last_activity_unix = max(ifnull(last_activity_unix, 0), ?) WHERE address = ?",
                           (tx_timestamp, fee_payer))
            if tx_data.get("type") == "SWAP":
                cursor.execute("UPDATE profiles SET swap_count = swap_count + 1 WHERE address = ?", (fee_payer,))
            if "pump.fun" in tx_data.get("source", "").lower():
                for inst in tx_data.get("transaction", {}).get("message", {}).get("instructions", []):
                    if inst.get("programId") == PUMP_FUN_PROGRAM_ID and len(inst.get("accounts", [])) > 6:
                        creator = inst["accounts"][6]
                        if creator == fee_payer: # Đảm bảo fee_payer chính là creator
                            cursor.execute("UPDATE profiles SET pump_fun_creations = pump_fun_creations + 1 WHERE address = ?", (creator,))
                            # Thêm token vào bảng created_tokens
                            token_mint = inst["accounts"][0]
                            cursor.execute("INSERT OR IGNORE INTO created_tokens (mint_address, creator_address, creation_date_unix, status) VALUES (?, ?, ?, ?)",
                                           (token_mint, creator, tx_timestamp, 'New'))
                            break # Đã tìm thấy, thoát vòng lặp instructions
        
        if (i + 1) % 500 == 0: conn.commit()
        
    conn.commit()
    print(f"\n✅ Xây dựng/cập nhật hồ sơ hoàn tất. Tổng cộng đã tạo {new_profiles_created} hồ sơ mới.")
    
    # --- Bước 3: Cập nhật vai trò tổng hợp ---
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
    
    # --- Bước 4: In Báo cáo Nhanh sau khi xử lý ---
    print("\n" + "="*70)
    print(" BÁO CÁO NHANH SAU KHI XỬ LÝ ")
    print("="*70)
    conn, cursor = connect_db()
    if conn:
        cursor.execute("SELECT COUNT(*) FROM profiles")
        total_profiles = cursor.fetchone()[0]
        print(f"📊 TỔNG SỐ HỒ SƠ TRONG DATABASE: {total_profiles}")
        cursor.execute("SELECT COUNT(*), roles FROM profiles GROUP BY roles")
        print("\n📊 PHÂN BỐ VAI TRÒ HIỆN TẠI:")
        for row in cursor.fetchall():
            print(f"  - {row['roles'] if row['roles'] else 'Chưa xác định'}: {row['COUNT(*)']} ví")
        conn.close()

    print("\n" + "="*70)
    print(" HOÀN TẤT PHÂN TÍCH. DATABASE ĐÃ SẴN SÀNG CHO REPORTER.PY. ")
    print("="*70)

if __name__ == "__main__":
    process_and_build_graph()