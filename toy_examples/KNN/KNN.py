import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# 加载数据
digits = load_digits()
data = digits['data']
# 数据探索
# print(data.shape)
# 查看第一幅图像
# print(digits['images'][0])
# 第一幅图像代表的数字含义
# print(digits['target'][0])
# 将第一幅图像显示出来
# plt.gray()
# plt.imshow(digits['images'][0])
# plt.show()
# 分割数据，将 25% 的数据作为测试集，其余作为训练集（你也可以指定其他比例的数据作为训练集）
train_x, test_x, train_y, test_y = train_test_split(data, digits['target'], test_size=0.25, random_state=33)
# 采用 Z-Score 规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
# 创建 KNN 分类器
knn = KNeighborsClassifier() 
knn.fit(train_ss_x, train_y) 
predict_y = knn.predict(test_ss_x) 
print("KNN 准确率: %.4lf" % accuracy_score(predict_y, test_y)) # 宽度为4的double型双精度
