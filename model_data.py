import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',12)


cars = pd.read_csv("/Users/ishanranasinghe/Desktop/Filtered_Car_Dataset.csv")

print(cars['make'].unique())

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