import pickle
import pandas as pd
pd.set_option('display.width',300)
pd.set_option('display.max_columns',12)
from sklearn.preprocessing import LabelEncoder

cars_df =  pd.read_csv("/Users/ishanranasinghe/Desktop/Filtered_Car_Dataset.csv")
print(cars_df.head(10))


lbl_encoder = LabelEncoder()
cars_df['make'] = lbl_encoder.fit_transform(cars_df['make'])
pickle.dump(lbl_encoder, open('make_col_enc.pkl','wb'))