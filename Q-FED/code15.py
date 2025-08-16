import asyncio
import json
import time
import requests
import websockets
from solders.pubkey import Pubkey

# ==============================================================================
# --- CẤU HÌNH ---
# ==============================================================================
# --- Helius API ---
# API Key của bạn (từ Helius)
HELIUS_API_KEY = "b7136879-e7d5-47f7-9aff-d2640767f758" 
HELIUS_WSS_URL = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
HELIUS_API_URL = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"

# --- Telegram Bot ---
# Token Bot Telegram của bạn
BOT_TOKEN = "8165514260:AAFvRf0TAQ0gdowH3AoKjNinsDjSUhSGlg8" 
# ID Kênh Telegram của bạn (nhớ cả dấu -)
CHANNEL_ID = "-1002877778546" 

# --- DANH SÁCH VÍ CẦN THEO DÕI ---
# Danh sách này cần được cập nhật thủ công (hoặc thông qua file log)
KNOWN_TEAM_WALLECTS = {
    "771zujoMQDHMGyBWu363HNHr3PPXbSkA7X5kiQBpPNSz": "Ví Gốc / Mẹ",
    "GnE2cQ455ZyvQcP9xfLoVEo4orhyhj8qBz5q1iW1r5nt": "Creator 1 ($BALLZ)",
    "7gbnR6f7EyTyMu6ECf4ndnwREG4TCqtruSvD6gEG4gex": "Creator 2",
    "H9bBXbLgkHrrfnZCvEpGBR4ekWaDeL5LjjkT7SFX7qWK": "Creator 3"
}

# File để lưu các ví mới phát hiện
NEW_WALLETS_LOG_FILE = "new_wallets.log"

# ==============================================================================
# --- MODULES ---
# ==============================================================================

def send_telegram_message(message_text: str):
    """Gửi tin nhắn đến kênh Telegram."""
    api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHANNEL_ID, 'text': message_text,
        'parse_mode': 'Markdown', 'disable_web_page_preview': True
    }
    try:
        requests.post(api_url, json=payload, timeout=10).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"    [!] Lỗi Telegram: {e}")

def log_new_wallet(wallet_address: str):
    """Lưu địa chỉ ví mới phát hiện vào file log."""
    # Logic kiểm tra và ghi log
    try:
        with open(NEW_WALLETS_LOG_FILE, "a+") as f:
            f.seek(0)
            existing_wallets = f.read().splitlines()
            if wallet_address not in existing_wallets and wallet_address not in KNOWN_TEAM_WALLECTS:
                f.write(wallet_address + "\n")
                print(f"  -> Đã lưu ví mới vào `{NEW_WALLETS_LOG_FILE}`: {wallet_address}")
    except IOError as e:
        print(f"Lỗi khi ghi file log: {e}")

def analyze_and_report(signature: str):
    """
    Hàm lõi: Phân tích signature và gửi báo cáo qua Telegram.
    """
    print(f"\n🔬 Đang phân tích signature: {signature}...")
    try:
        response = requests.post(HELIUS_API_URL, json={"transactions": [signature]})
        response.raise_for_status()
        tx_data = response.json()[0]
    except Exception as e:
        print(f"Lỗi khi phân tích TX: {e}")
        error_message = f"🚨 *LỖI PHÂN TÍCH GIAO DỊCH* 🚨\n\n`{signature}`\nLỗi: `{e}`"
        send_telegram_message(error_message)
        return

    # Xác định ví nào trong team có liên quan
    involved_team_wallets = []
    
    # Duyệt qua tất cả các tài khoản tham gia giao dịch và kiểm tra
    for account in tx_data.get("accountData", []):
        if account["account"] in KNOWN_TEAM_WALLECTS:
            involved_team_wallets.append(account["account"])

    if not involved_team_wallets:
        return

    # Xây dựng báo cáo
    tx_type = tx_data.get("type", "UNKNOWN")
    description = tx_data.get("description", "")
    source = tx_data.get("source", "N/A")
    
    # Header báo cáo
    wallet_names = ", ".join([KNOWN_TEAM_WALLECTS.get(w, w[:4]+'...') for w in set(involved_team_wallets)])
    report = [f"🚨 *HOẠT ĐỘNG MỚI TỪ TEAM [{wallet_names}]* 🚨\n"]
    report.append(f"**Loại:** `{tx_type}`")
    if description:
        report.append(f"**Mô tả:** {description}")

    # Phân tích sâu hơn cho các loại giao dịch
    if tx_type == "TRANSFER":
        events = tx_data.get("events", {}).get("nativeTransfers", [])
        for event in events:
            sender = event['fromUserAccount']
            receiver = event['toUserAccount']
            amount = event['amount'] / 1_000_000_000
            
            # Chỉ tập trung vào các giao dịch của team
            if sender in KNOWN_TEAM_WALLECTS:
                report.append(f"  - `{KNOWN_TEAM_WALLECTS[sender]}` đã gửi `{amount:.4f} SOL` đến `{receiver[:4]}...{receiver[-4:]}`")
                
                # Phát hiện ví mới
                if receiver not in KNOWN_TEAM_WALLECTS:
                    report.append(f"  - 🕵️‍♂️ *VÍ MỚI?* `{receiver}`")
                    log_new_wallet(receiver)
    
    elif "pump.fun" in source.lower() or tx_type == "UNKNOWN":
        # Xác định giao dịch tạo token
        instructions = tx_data.get("transaction", {}).get("message", {}).get("instructions", [])
        
        # Địa chỉ chương trình Pump.fun
        pump_fun_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

        for inst in instructions:
            if inst.get("programId") == pump_fun_program_id:
                 accounts = inst.get("accounts", [])
                 if len(accounts) > 6:
                    token_mint = accounts[0]
                    creator = accounts[6]
                    if creator in KNOWN_TEAM_WALLECTS:
                        report.append(f"  - 💎 *TẠO TOKEN MỚI!*")
                        report.append(f"  - Creator: `{KNOWN_TEAM_WALLECTS[creator]}`")
                        report.append(f"  - Token Mint: `{token_mint}`")
                        report.append(f"  - [Xem trên Pump.fun](https://pump.fun/{token_mint})")
    
    # Thêm các loại giao dịch khác nếu cần (SWAP, NFT,...)

    # Footer báo cáo
    report.append(f"\n[Xem giao dịch trên Solscan](https://solscan.io/tx/{signature})")
    
    # Gửi báo cáo hoàn chỉnh
    final_report = "\n".join(report)
    print("✅ Phân tích xong, đang gửi thông báo...")
    send_telegram_message(final_report)


# ==============================================================================
# --- HÀM MAIN (LẮNG NGHE REAL-TIME) ---
# ==============================================================================
async def main():
    print("🚀 Bắt đầu chạy Whale Watcher Pro V2.0...")
    
    async with websockets.connect(HELIUS_WSS_URL) as ws:
        # Lấy danh sách ví cần theo dõi
        wallet_list = list(KNOWN_TEAM_WALLECTS.keys())
        
        # Yêu cầu lắng nghe log giao dịch có nhắc đến các ví này
        subscribe_request = {
            "jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
            "params": [{"mentions": wallet_list}, {"commitment": "confirmed"}]
        }
        await ws.send(json.dumps(subscribe_request))
        print(f"✅ Đã đăng ký theo dõi {len(wallet_list)} ví. Đang lắng nghe real-time...")
        
        async for message in ws:
            try:
                data = json.loads(message)
                if 'params' in data and 'result' in data['params']:
                    signature = data['params']['result']['value']['signature']
                    if signature:
                        # Gọi hàm phân tích cho mỗi signature mới nhận được
                        analyze_and_report(signature)
                        
            except Exception as e:
                print(f"--- Lỗi khi xử lý message: {e} ---")

if __name__ == "__main__":
    # Khởi tạo và xử lý lỗi kết nối
    while True:
        try:
            asyncio.run(main())
        except websockets.exceptions.ConnectionClosed:
            print("🛑 Kết nối bị đóng. Đang thử kết nối lại sau 5 giây...")
            time.sleep(5)
        except Exception as e:
            print(f"Lỗi không xác định: {e}. Đang khởi động lại sau 10 giây...")
            time.sleep(10)