import numpy as np

# hàm chia lá
def split_node(column, threshold_split):  # column là series
    left_node = column[column <= threshold_split]  # series chứa những phần tử nhỏ hơn hoặc bằng ngưỡng
    right_node = column[column > threshold_split]  # series chứa những phần tử lớn hơn ngưỡng
    return left_node, right_node

# hàm entropy
def entropy(y_target):  # y_target là dạng series
    values, counts = np.unique(y_target, return_counts=True)
    result = -np.sum([i * np.log2(i) for i in counts / len(y_target)])
    return result  # một con số

# hàm information gain
def info_gain(column, target, threshold_split):  # column, target là series
    entropy_first = entropy(target)  # entropy ban đầu

    left_node, right_node = split_node(column, threshold_split)

    n_target, n_left, n_right = len(target), len(left_node), len(right_node)

    entropy_left = entropy(target[left_node.index])
    entropy_right = entropy(target[right_node.index])

    sum_entropy_node = (n_left / n_target) * entropy_left + (n_right / n_target) * entropy_right

    ig = entropy_first - sum_entropy_node  # information gain

    return ig

# hàm best_feature, best_threshold
def best_split(dataX, target):  # dataX dạng Frame, target dạng series
    best_ig = -np.inf  # ban đầu cho bằng giá trị âm vô cùng để so sánh
    best_feature = None
    best_threshold = None
    for c in dataX.columns:
        column = dataX[c]  # series
        thresholds = set(column)  # lấy unique
        for threshold in thresholds:  # các phần tử số thì threshold lưu dạng số
            ig = info_gain(column, target, threshold)
            if ig > best_ig:
                best_ig = ig
                best_feature = c
                best_threshold = threshold
    return best_feature, best_threshold  # best_feature là tên cột, best_threshold là số

# hàm lấy giá trị có số lượng nhiều nhất trong lá
def most_value(y_target):  # y_target là series
    value = y_target.value_counts(sort=True).index[0]
    return value

# class Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # chọn feature tốt nhất để chia, khi node này là node lá thì feature = None
        self.threshold = threshold  # ngưỡng chia tốt nhất
        self.left = left  # nút con bên trái
        self.right = right  # nút con bên phải
        self.value = value  # giá trị nút lá

    def is_leaf_node(self):
        return self.value is not None  # nếu giá trị nút lá không phải none thì return True

# lớp Decision Tree Classification
class DecisionTreeClass:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split  # số lượng mẫu tối thiểu để chia một nút
        self.max_depth = max_depth  # độ sâu của cây 
        self.root = None  # khởi tạo gốc cây là None

    def grow_tree(self, X, y, depth=0):  # X là frame, y là series
        n_y, n_feature = X.shape
        n_class = len(set(y))

        if depth >= self.max_depth or n_class == 1 or n_y < self.min_samples_split:
            value_leaf = most_value(y)
            return Node(value=value_leaf)

        best_feature, best_threshold = best_split(X, y)

        # tạo node con
        left_node, right_node = split_node(X[best_feature], best_threshold)

        left = self.grow_tree(X.iloc[left_node.index, :], y[left_node.index], depth + 1)
        right = self.grow_tree(X.iloc[right_node.index, :], y[right_node.index], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):  # X là frame, y là series
        self.root = self.grow_tree(X, y)

    # tạo hàm duyệt cây
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for _, x in X.iterrows()])
    
    
import pandas as pd

# Dữ liệu thời tiết
data = {
    'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 91],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Chuyển đổi sang DataFrame
df = pd.DataFrame(data)

# Encode các cột 'Outlook' và 'Windy' (categorical) thành dạng số
df['Outlook'] = df['Outlook'].map({'sunny': 0, 'overcast': 1, 'rainy': 2})
df['Windy'] = df['Windy'].astype(int)

# Chia dữ liệu thành X và y
X = df.drop('Play', axis=1)  # Đầu vào (input)
y = df['Play'].map({'No': 0, 'Yes': 1})  # Mục tiêu (output) - Encode 'No' thành 0 và 'Yes' thành 1

# Khởi tạo và huấn luyện mô hình
tree = DecisionTreeClass(min_samples_split=2, max_depth=3)
tree.fit(X, y)

# Dự đoán trên dữ liệu ban đầu
predictions = tree.predict(X)

# In kết quả dự đoán và kết quả thực tế
print("Dự đoán: ", predictions)
print("Thực tế:  ", y.values)
