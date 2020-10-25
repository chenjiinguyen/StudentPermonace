import numpy
import pandas
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pandas.read_csv('student-mat.csv', low_memory=False)

# Hiệu Chỉnh Data và Xét trong khoảng điều kiện

## sex - giới tính (được recode lại thành mã nhị phâm)
data['sex'] = data['sex'].map({"M":0 ,"F":1})
## absences - số lần nghỉ học
data['absences'] = pandas.to_numeric(data['absences'], errors='coerce')
## G3 - số điểm cuối cấp
data['G3'] = pandas.to_numeric(data['G3'], errors='coerce')
## Xét dữ liệu trong điều kiện
new_data = data[(data['age']>=16) & (data['age']<=22)]

# Phân tích Hồi Quy hai biến định lượng
print("Phân tích Hồi Quy hai biến định lượng")
result = smf.ols('absences ~ G3', data=data).fit()
print (result.summary())

# Đồ thị Phân tích Hồi Quy hai biến định lượng
seaborn.regplot(x="G3", y="absences", data=new_data)
plt.xlabel('Điểm cuối cấp')
plt.ylabel('Số lần nghỉ học')
plt.title('Đồ thị biểu diễn 2 biến định lượng')
plt.show()

# Kết Quả Phân Tích:
# - Phương trình: absences = 0.0598 * G3 - 5.0858
# - Từ phương trình ta thấy rằng Điểm Cuối Cấp và Số Lần Nghỉ Học 
# liên quan rất ít đến nhau. Vui lòng xem đồ thị để thấy đường thẳng 
# được vẽ từ phương trình trên.



# Phân tích hồi quy với biến giải thích là phân loại
print("Phân tích hồi quy với biến giải thích là phân loại")
result = smf.ols('absences ~ sex', data=data).fit()
print (result.summary())

# Đồ thị Phân tích hồi quy với biến giải thích là phân loại
seaborn.factorplot(x="sex", y="absences", data=new_data, kind="bar", ci=None)
plt.xlabel('Giới tính')
plt.ylabel('Số lần nghỉ học')
plt.title('Đồ thị biểu diễn với biến giải thích là phân loại')
plt.show()

# Kết Quả Phân Tích:
# - Phương trình: absences = 1.0720 * sex + 5.1444
# - Theo mục recode thì nếu giới tích là nam thì sẽ là 0 
# còn nữ thì sẽ là 1. Vì thế, ta có thẻ tìm được số lần 
# vắng học theo giới tính là 0 và 1 bằng cách dùng phương trình trên
# + Số lần vắng học của nam: 1.0720 * 0 + 5.1444 = 5.1444 lần
# + Số lần vắng học của nữ: 1.0720 * 1 + 5.1444 = 6.2164 lần
# - Kết luận: Những bạn nam sẽ có trung bình 5.1444 lần vắng học.
# Còn số lần vắng học của các bạn nữ sẽ là 6.2164 lần.