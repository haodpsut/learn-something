import requests
import json
import time
import sqlite3
import os

# ==============================================================================
# --- CẤU HÌNH ---
# ==============================================================================
HELIUS_API_KEYS = [
    # !!! DÁN DANH SÁCH API KEY CỦA BẠN VÀO ĐÂY (ít nhất 1 key) !!!
    "9c0bdddf-d974-48b0-a075-548e8cea1ea0",
    "f0bf36da-0aa2-417f-a38a-71e1ee904263",
    # "key_2...",
]
current_api_key_index = 0

DATABASE_FILE = "profiles.db"
PUMP_FUN_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

ENRICHMENT_BATCH_SIZE = 20 # Mỗi lần chạy chỉ làm giàu 20 hồ sơ
HISTORY_LIMIT_PER_WALLET = 200 # Chỉ lấy 200 giao dịch gần nhất của mỗi ví vệ tinh

# ==============================================================================
# --- MODULES ---
# ==============================================================================

def get_rpc_endpoint():
    """Lấy RPC endpoint hiện tại từ danh sách xoay vòng."""
    return f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEYS[current_api_key_index]}"

def get_das_api_url():
    """Lấy URL của DAS API hiện tại từ danh sách xoay vòng."""
    return f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEYS[current_api_key_index]}"

def rotate_api_key():
    """Xoay vòng sang API key tiếp theo."""
    global current_api_key_index
    previous_key_index = current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(HELIUS_API_KEYS)
    print(f"\n[!] Hạn ngạch API Key #{previous_key_index + 1} có thể đã hết. Chuyển sang Key #{current_api_key_index + 1}...")
    time.sleep(5)
    return current_api_key_index == 0

def connect_db():
    """Tạo kết nối đến database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    except sqlite3.Error as e:
        print(f"Lỗi khi kết nối DB: {e}")
        return None, None

def get_signatures_for_address(wallet_address: str, limit: int):
    """Lấy chữ ký giao dịch gần nhất của một ví."""
    try:
        payload = {"jsonrpc": "2.0", "id": "1", "method": "getSignaturesForAddress", "params": [wallet_address, {"limit": limit}]}
        response = requests.post(get_rpc_endpoint(), json=payload, timeout=30)
        response.raise_for_status()
        return [tx['signature'] for tx in response.json().get('result', [])]
    except Exception as e:
        print(f"    Lỗi khi lấy chữ ký: {e}")
        # Thử xoay vòng key nếu gặp lỗi 429
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
            rotate_api_key()
        return []

def get_transactions_details_resilient(signatures: list):
    """Lấy chi tiết giao dịch một cách bền bỉ, có xoay vòng API key."""
    if not signatures: return []
    all_tx_details = []
    batch_size = 40
    i = 0
    while i < len(signatures):
        batch = signatures[i:i + batch_size]
        try:
            response = requests.post(get_das_api_url(), json={"transactions": batch}, timeout=120)
            if response.status_code == 429:
                if rotate_api_key(): break
                continue
            response.raise_for_status()
            all_tx_details.extend(response.json())
            i += batch_size
            time.sleep(1.5)
        except requests.exceptions.RequestException as e:
            print(f"\nLỗi mạng: {e}. Dừng lại."); break
    return all_tx_details

# ==============================================================================
# --- HÀM CHÍNH ---
# ==============================================================================
def enrich_profiles_final():
    conn, cursor = connect_db()
    if not conn: return

    print("="*70); print(" BẮT ĐẦU LÀM GIÀU HỒ SƠ (ENRICHER V1.1 - SỬA LỖI TRẠNG THÁI) "); print("="*70)

    # --- Bước 1: Lấy danh sách hồ sơ CHƯA ĐƯỢC LÀM GIÀU ---
    # Logic mới: Dựa vào cột `total_tx`. total_tx = 0 là trạng thái ban đầu.
    cursor.execute("""
        SELECT address, name FROM profiles
        WHERE total_tx = 0
        LIMIT ?
    """, (ENRICHMENT_BATCH_SIZE,))

    wallets_to_enrich = cursor.fetchall()

    if not wallets_to_enrich:
        print("✅ Không còn hồ sơ nào cần làm giàu ban đầu (total_tx = 0). Quá trình hoàn tất."); conn.close(); return

    print(f"Sẽ làm giàu cho {len(wallets_to_enrich)} hồ sơ trong lần chạy này...")

    # Lấy tiến độ tổng thể để báo cáo
    cursor.execute("SELECT count(*) FROM profiles WHERE total_tx != 0")
    total_processed_wallets = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM profiles")
    total_wallets = cursor.fetchone()[0]

    for i, wallet in enumerate(wallets_to_enrich):
        wallet_address = wallet['address']
        print(f"\n--- [{i+1}/{len(wallets_to_enrich)}] Đang làm giàu cho: {wallet['name']} ({wallet_address[:6]}...) ---")

        signatures = get_signatures_for_address(wallet_address, limit=HISTORY_LIMIT_PER_WALLET)

        if not signatures:
            # Đặt total_tx = -1 để đánh dấu là "Đã kiểm tra, không có giao dịch"
            cursor.execute("UPDATE profiles SET total_tx = -1 WHERE address = ?", (wallet_address,))
            conn.commit()
            print("  -> Không có giao dịch nào, đánh dấu đã xử lý.")
            continue

        tx_details = get_transactions_details_resilient(signatures)

        swap_count = 0
        pump_fun_creations = 0
        for tx in tx_details:
            if tx.get("type") == "SWAP": swap_count += 1
            if "pump.fun" in tx.get("source", "").lower():
                for inst in tx.get("transaction", {}).get("message", {}).get("instructions", []):
                    if inst.get("programId") == PUMP_FUN_PROGRAM_ID and len(inst.get("accounts", [])) > 6:
                        if inst["accounts"][6] == wallet_address:
                            pump_fun_creations += 1; break

        # Cập nhật vào DB
        total_tx_found = len(tx_details)
        cursor.execute("""
            UPDATE profiles
            SET
                total_tx = ?,
                swap_count = ?,
                pump_fun_creations = ?
            WHERE address = ?
        """, (total_tx_found, swap_count, pump_fun_creations, wallet_address))

        conn.commit()
        print(f"  -> ✅ Đã cập nhật hồ sơ: {total_tx_found} tx, {swap_count} swaps, {pump_fun_creations} creations.")

    conn.close()

    print("\n" + "="*70)
    print(" HOÀN TẤT LẦN CHẠY LÀM GIÀU. ")
    print(f"  -> Tiến độ tổng thể: {total_processed_wallets + len(wallets_to_enrich)} / {total_wallets} hồ sơ đã được làm giàu.")
    print("  -> Hãy chạy lại script để tiếp tục.")
    print("="*70)

if __name__ == "__main__":
    # --- Điều kiện kiểm tra mới, thông minh hơn ---
    # Kiểm tra xem danh sách có rỗng không và key đầu tiên có vẻ hợp lệ không
    is_config_ok = True
    if not HELIUS_API_KEYS:
        is_config_ok = False
    elif len(HELIUS_API_KEYS[0]) < 36 or '-' not in HELIUS_API_KEYS[0]: # Key Helius thường có dạng UUID dài 36 ký tự
        is_config_ok = False
        print("!!! CẢNH BÁO: API Key đầu tiên có vẻ không hợp lệ. Vui lòng kiểm tra lại. !!!")

    if not is_config_ok:
        print("!!! LỖI CẤU HÌNH: Vui lòng mở file và điền các API Key thật vào danh sách `HELIUS_API_KEYS`. !!!")
    else:
        enrich_profiles_final()