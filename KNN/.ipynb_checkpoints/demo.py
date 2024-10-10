import matplotlib.pyplot as plt

# Dữ liệu để vẽ
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Vẽ biểu đồ đường
plt.plot(x, y, label="Biểu đồ đường", color='b', marker='o')

# Thêm tiêu đề và nhãn cho các trục
plt.title("Biểu đồ Đường Đơn Giản")
plt.xlabel("Trục X")
plt.ylabel("Trục Y")

# Hiển thị chú thích
plt.legend()

# Hiển thị biểu đồ
plt.show()
