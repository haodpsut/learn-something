def is_prime(n):
    """Kiểm tra số nguyên tố."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Ví dụ sử dụng
if __name__ == "__main__":
    for num in range(1, 21):
        if is_prime(num):
            print(f"{num} là số nguyên tố")