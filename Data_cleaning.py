import numpy as np # Multi-dimensional array object
import pandas as pd # Data Manipulation

import matplotlib.pyplot as plt

pd.set_option('display.width',300)
pd.set_option('display.max_columns',12)

cars_data = pd.read_csv('/Users/ishanranasinghe/Desktop/Filtered_Car_Dataset.csv')
print(cars_data.head(10))

cars_data["price"] = cars_data["price"].str.replace("Rs.", "")
cars_data["price"] = cars_data["price"].str.replace(",", "")
cars_data["price"] = cars_data["price"].astype(int)

cars_data["price"] = cars_data["price"].str.replace("Negotiable","")
cars_data["price"] = cars_data["price"].astype(int)

cars_data["mileage"] = cars_data["mileage"].str.replace("-","")
cars_data["mileage"] = cars_data["mileage"].astype(int)

cars_data["engine_capacity"] = cars_data["engine_capacity"].str.replace("-","")
cars_data["engine_capacity"] = cars_data["engine_capacity"].astype(int)

# drop unwanted rows
df_new = cars_data.drop(cars_data[(cars_data['col_1'] == 1.0) & (cars_data['col_2'] == 0.0)].index)

pd.set_option('display.width',300)
pd.set_option('display.max_columns',8)
cars = pd.read_csv("Data_set/cars_data.csv")
# print(cars.shape)
# print(cars.head(100))
# print(cars.isnull().sum())

# plt.boxplot(cars['price'])
# plt.show()

x = cars['model_year']
y = cars['price']
plt.scatter(x,y, label ="model_year price")
plt.show()


