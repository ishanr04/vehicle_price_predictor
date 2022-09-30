import pickle
import pandas as pd
import numpy as np
import seaborn as sns # Data Visualization
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


cars = pd.read_csv("/Users/ishanranasinghe/Desktop/Filtered_Car_Dataset.csv")


make_col_enc = pickle.load(open('Data_Encoding/make_col_enc.pkl','rb'))
cars['make'] = make_col_enc.transform(cars['make'])

model_col_enc = pickle.load(open('Data_Encoding/model_col_enc.pkl','rb'))
cars['model'] = model_col_enc.transform(cars['model'])


cars_pre = pd.get_dummies(cars, columns=['transmission','fuel_type'])
X = cars_pre.drop('price',axis=1)
Y = cars_pre['price']

a_pre = np.array(X)
b_pre = np.array(Y)

a_pre_train , a_pre_test, b_pre_train, b_pre_test = train_test_split(a_pre,b_pre, test_size= 0.2)

RandomForest_model = RandomForestRegressor()
RandomForest_model.fit(a_pre_train,b_pre_train)

accuracy_train = RandomForest_model.score(a_pre_train,b_pre_train)
print('Training Accuracy:',accuracy_train*100,'%')

b_train_predict = RandomForest_model.predict(a_pre_train)
print('MAE:', mean_absolute_error(b_pre_train, b_train_predict))
print('R2 score', r2_score(b_pre_train, b_train_predict))


print("-----------------------------")

accuracy_RandomForest = RandomForest_model.score(a_pre_test,b_pre_test)
print('Testing Accuracy',accuracy_RandomForest*100,'%')

b_predict = RandomForest_model.predict(a_pre_test)
print('MAE:', mean_absolute_error(b_pre_test, b_predict))
print('R2 score', r2_score(b_pre_test, b_predict))


fig = sns.regplot(x=b_predict,y=b_pre_test,color='blue', marker="^")
fig.set(title="Random Forest Model", xlabel = "Predicted price")
plt.show()
