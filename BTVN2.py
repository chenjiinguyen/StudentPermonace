import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt


# Đọc dữ liệu
data = pandas.read_csv('student-mat.csv', low_memory=False)

# Xem xét kích thước dữ liệu
# print(len(data))
# print(len(data.columns))

# Sao Chep Du Lieu Moi
new_data = data.copy()

# Bỏ Các Dữ Liệu Thiếu
new_data["failures"] = new_data["failures"].replace(4, numpy.nan)

# Chuyển Đổi Codebook
new_data["failures"] = new_data["failures"].map({0:"0.Not yet",1:"1.Less",2:"2.Medium",3:"3.Many"})
new_data["activities"] = new_data["activities"].map({"yes":"Yes","no":"No"})
new_data["goout"] = new_data["goout"].map({1:"1.Very Low",2:"2.Low",3:"3.Normal",4:"4.High",5:"5.Very High"})
# In Thống Kê

## failures 
c1 = new_data["failures"].value_counts(dropna=False, normalize=True).sort_index()
print("Phần Trăm Ý Kiến Học Sinh Ghi Nhận Thất Bại Của Mình")
print(c1)

print("Thống Kê Ý Kiến Học Sinh Ghi Nhận Thất Bại Của Mình")
c2 = new_data["failures"].describe()
print(c2)

print("")
## activities
c3 = new_data["activities"].value_counts(dropna=False).sort_index()
print("Phần Trăm Ý Kiến Của Học Sinh Về Việc Tham Gia Hoạt Động Ngoại Khóa")
print(c3)

print("Thống Kê Ý Kiến Của Học Sinh Về Việc Tham Gia Hoạt Động Ngoại Khóa")
c4 = new_data["activities"].describe()
print(c4)

print("")
## goout
c5 = new_data["goout"].value_counts(dropna=False, normalize=True).sort_index()
print("Phần Trăm Ý Kiến Của Học Sinh Về Việc Đi Chơi Với Bạn Bè")
print(c5)

print("Thống Kê Ý Kiến Của Học Sinh Về Việc Đi Chơi Với Bạn Bè")
c6 = new_data["goout"].describe()
print(c6)

# In Biểu Đồ

## failures
# seaborn.countplot(x="failures",
#                 data=new_data)
# plt.title("Biểu Đồ Ý Kiến Học Sinh Ghi Nhận Thất Bại Của Mình")
# plt.xlabel('View')
# plt.ylabel('Số Lượng')
# plt.show()
## activities
# seaborn.countplot(x="activities",
#                 data=new_data)
# plt.title("Biểu Đồ Ý Kiến Của Học Sinh Về Việc Tham Gia Hoạt Động Ngoại Khóa")
# plt.xlabel('View')
# plt.ylabel('Số Lượng')
# plt.show()
## activities
seaborn.countplot(x="goout",
                data=new_data)
plt.title("Biểu Đồ Ý Kiến Của Học Sinh Về Việc Đi Chơi Với Bạn Bè")
plt.xlabel('View')
plt.ylabel('Số Lượng')
plt.show()