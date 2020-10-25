import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

# Đọc dữ liệu
data = pandas.read_csv('student-mat.csv', low_memory=False)

# Sao Chep & Hieu Chiu Du Lieu Moi
new_data = data.copy()
new_data['internet_numberic'] = new_data['internet'].map({'yes':1,'no':0})
new_data['internet_numberic'] = pandas.to_numeric(new_data['internet_numberic'], errors='coerce')


# ANOVA Cho Bien Giai Thich Co 2 Loai
model1 = smf.ols(formula='G3 ~ C(internet_numberic)', data=new_data)
results1 = model1.fit()
print(results1.summary())


