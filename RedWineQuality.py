import numpy as np
import matplotlib.pyplot  as plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


docdata = pd.read_csv ("/media/iustin/500CD4800CD46310/Iustin/RedWineQuality/winequality-red.csv")
df = docdata.copy ()
# print (df.head ())
# print (df.info())
df.drop_duplicates (inplace = True)
df.reset_index (drop = True, inplace = True)
# print( df.shape)
# print (df.info())
missing_data = df.isnull ()
# print (missing_data.sum ())

# df.plot()
# sns.pairplot(df, diag_kind='hist', corner=True)
# hmap_mask = np.triu(df.corr(), k=1)
# plt.rc('font', size=10)
# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(), mask=hmap_mask, annot = True, fmt='.2f', cmap='coolwarm', annot_kws={"fontsize":10})
# plt.show()

"Splitiing data into features and labels"
x = df.drop('quality', axis = 1)
y = df.iloc[:,-1:]

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='  ')