from collections import deque

# Tạo một queue rỗng
queue = deque()

# Thêm phần tử vào queue (enqueue)
queue.append('A')
queue.append('B')
queue.append('C')

# Lấy phần tử ra khỏi queue (dequeue)
item = queue.popleft()
print(f"Phần tử lấy ra: {item}")

# Hiển thị queue hiện tại
print("Queue hiện tại:", list(queue))