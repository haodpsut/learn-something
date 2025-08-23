# main.py
from database_manager import DatabaseManager
from crawler import SolanaCrawler

def main():
    # === CẤU HÌNH ===
    # Thêm các ví gốc bạn muốn bắt đầu theo dõi vào đây
    ROOT_WALLETS = [
    "HcRDCCdT1u28Ez4QjXjze7Su5HkNftrFN6YYpEi9n8ao",
    "EqQ59bgANjo1DzKKXksXa6pVxsy3jmMmDty7ar9pQ7s",
    "HUm9aFr6zGEVPrJMAT3V78tiwGeZdj4xHrm1NmC2qBji",
    "2PRiH18FbyPKRVo7tcCwdwuYEbawQsq6HWmGFVb4Tz5R",
    "4RPa4q9ABqQs5dXmyKqZkDQ6qTZPe2ipYe2Cpa37nKM6",
    "6W5cs1NNUrMcNDdopE8efnJqBF8wWprmK7JjsCLf5DPX",
    "8XmxucwDcEeVQncjvfjfok4kAnjfAaWifNkzQbJmeu8m",
    "6rjJzEjocA73udX5p6p7p9HJeDuSfLdHkLJdHnwvGxEc",
    "A5myrfFU5LBN1YTtfJePrzQc289hgg9zJUGDUCDjRvHw",
    "6V4TNFisbWS6Yr7wTa2C2GeXLoUeNw4oC55WpPWSD92K",
    "4auqT1LJcpXRGzpgTHkb2vBVNzmAotiWsHPk3Z8rDEQF",
    "GvNVqRUD85YVTGyuQ5jK5PJ6PX66NE8FNX25Nyguoz8S",
    "noz6uoYCDijhu1V7cutCpwxNiSovEwLdRHPwmgCGDNo",
    "6mEZTqrAjQjVFjSCjk2BeuyQoF8WMHJtNJahATBhzbyX",
    "GdZJRqonwzeWRneFVrniwU5PeQqJa7bB8G12qdQqhU1H",
    "E7j8oEfgh9NuZNzZvRRT8rXkgRy4WXc4sVkGvEAJa7Wt"
        # Thêm các ví khác nếu có
    ]

    # Khởi tạo các đối tượng
    db_manager = DatabaseManager("solana_wallets.db")
    crawler = SolanaCrawler(db_manager)

    # Thiết lập database
    db_manager.setup_database()

    # Thêm các ví gốc vào database
    for wallet in ROOT_WALLETS:
        if "ENTER_YOUR" in wallet: # Bỏ qua ví mẫu
            continue
        db_manager.add_wallet(wallet, is_root=True, status='pending')
    print(f"Đã thêm {len(ROOT_WALLETS)} ví gốc vào hàng chờ.")

    # Vòng lặp crawl chính
    while True:
        wallet_to_crawl = db_manager.get_pending_wallet()
        if wallet_to_crawl is None:
            print("\n===================================")
            print("Đã crawl xong tất cả các ví trong hàng chờ. Chương trình kết thúc.")
            print("===================================")
            break
        
        try:
            crawler.crawl_wallet(wallet_to_crawl)
        except Exception as e:
            print(f"Lỗi nghiêm trọng khi crawl ví {wallet_to_crawl}: {e}")
            db_manager.update_wallet_status(wallet_to_crawl, 'error')
    
    # Đóng kết nối database khi hoàn tất
    db_manager.close()


if __name__ == "__main__":
    main()