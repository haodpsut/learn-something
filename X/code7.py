import requests
import json
import sqlite3
import time
from solders.pubkey import Pubkey
from solders.account import Account

# ==============================================================================
# --- CẤU HÌ-NH ---
# ==============================================================================
HELIUS_RPC_ENDPOINT = "https://mainnet.helius-rpc.com/?api-key=9c0bdddf-d974-48b0-a075-548e8cea1ea0"
DATABASE_FILE = "profiles.db"

# === TOKEN CẦN QUÉT ===
# Dán địa chỉ mint của token bạn muốn điều tra vào đây
TOKEN_MINT_TO_SCAN = "Ce2gx9KGXJ6C9Mp5b5x1sn9Mg87JwEbrQby4Zqo3pump" # Ví dụ: $WIF
TOKEN_SYMBOL = "$neet" # Tên để hiển thị trong báo cáo

# ==============================================================================
# --- MODULES ---
# ==============================================================================

def connect_db():
    try:
        conn = sqlite3.connect(f"file:{DATABASE_FILE}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    except sqlite3.Error as e:
        print(f"Lỗi khi kết nối DB: {e}")
        return None, None

def get_associated_token_address(owner: Pubkey, mint: Pubkey) -> Pubkey:
    """Tính toán địa chỉ Associated Token Account (ATA)."""
    TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
    ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
    
    address, _ = Pubkey.find_program_address(
        [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID
    )
    return address

# ==============================================================================
# --- HÀM CHÍNH ---
# ==============================================================================
def scan_token_for_insider_holders():
    conn, cursor = connect_db()
    if not conn: return
    
    print("="*70)
    print(f" BẮT ĐẦU QUÉT INSIDER HOLDERS CHO TOKEN: {TOKEN_SYMBOL} ")
    print(f" Mint Address: {TOKEN_MINT_TO_SCAN}")
    print("="*70)

    # --- Bước 1: Tải "Bản đồ Mạng lưới" ---
    print("Đang tải 'Bản đồ Mạng lưới' vào bộ nhớ...")
    cursor.execute("SELECT address FROM profiles")
    team_wallets = [row['address'] for row in cursor.fetchall()]
    conn.close()
    
    if not team_wallets:
        print("Không tìm thấy hồ sơ nào trong database."); return
        
    print(f"✅ Đã tải {len(team_wallets)} ví 'người nhà'. Bắt đầu quét...")

    # --- Bước 2: Tính toán các địa chỉ ATA cần kiểm tra ---
    token_mint_pubkey = Pubkey.from_string(TOKEN_MINT_TO_SCAN)
    ata_addresses_to_check = []
    owner_map = {} # Dùng để map ATA về lại ví owner
    for owner_address in team_wallets:
        try:
            owner_pubkey = Pubkey.from_string(owner_address)
            ata_address = get_associated_token_address(owner_pubkey, token_mint_pubkey)
            ata_addresses_to_check.append(str(ata_address))
            owner_map[str(ata_address)] = owner_address
        except Exception:
            continue # Bỏ qua nếu địa chỉ ví không hợp lệ
            
    # --- Bước 3: Quét theo lô bằng getMultipleAccounts ---
    insider_holders = []
    batch_size = 100 # getMultipleAccounts có thể xử lý tối đa 100 tài khoản mỗi lần
    
    for i in range(0, len(ata_addresses_to_check), batch_size):
        batch = ata_addresses_to_check[i:i + batch_size]
        progress = ((i + batch_size) / len(ata_addresses_to_check)) * 100
        print(f"  -> Đang quét lô {i//batch_size + 1}/{ -(-len(ata_addresses_to_check)//batch_size)}... ({progress:.1f}%)", end='\r')
        
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1, 
                "method": "getMultipleAccounts",
                "params": [batch, {"encoding": "jsonParsed"}]
            }
            response = requests.post(HELIUS_RPC_ENDPOINT, json=payload, timeout=60)
            response.raise_for_status()
            
            results = response.json().get('result', {}).get('value', [])
            
            for idx, account_data in enumerate(results):
                if account_data: # Nếu tài khoản tồn tại
                    # Giải mã số dư
                    balance = int(account_data['data']['parsed']['info']['tokenAmount']['amount'])
                    if balance > 0:
                        ata_address = batch[idx]
                        owner_wallet = owner_map[ata_address]
                        insider_holders.append(owner_wallet)
            time.sleep(1) # Nghỉ giữa các lô
        except Exception as e:
            print(f"\nLỗi khi quét lô: {e}")
            time.sleep(5)
    
    print("\nQuét hoàn tất.")
    
    # --- Bước 4: Báo cáo kết quả ---
    print("\n" + "="*70)
    print(" BÁO CÁO KẾT QUẢ QUÉT ")
    print("="*70)
    
    if insider_holders:
        print(f"🔥 KẾT QUẢ: Tìm thấy {len(insider_holders)} ví 'người nhà' đang nắm giữ token {TOKEN_SYMBOL}!")
        print("   Danh sách (hiển thị tối đa 20):")
        for wallet in insider_holders[:20]:
            print(f"    - {wallet}")
    else:
        print(f"✅ KẾT QUẢ: Không tìm thấy ví 'người nhà' nào đang nắm giữ token {TOKEN_SYMBOL}.")

if __name__ == "__main__":
    scan_token_for_insider_holders()