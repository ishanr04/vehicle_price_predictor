import json

from flask import Flask, request, session, redirect, url_for, render_template
from flaskext.mysql import MySQL
import pymysql
import re
import mysql.connector
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',12)
# from flask_mysqldb import MySQL
#
application = Flask(__name__, template_folder='Template', static_folder='Assets')

import pymysql

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    db="fyp_db",
    cursorclass=pymysql.cursors.DictCursor
)


# application.config['MYSQL_HOST'] = 'localhost'
# application.config['MYSQL_USER'] = 'root'
# application.config['MYSQL_PASSWORD'] = ''
# application.config['MYSQL_DB'] = 'fyp_db'
#
# mysql = MySQL(application)

# mysql.init_app(application)

@application.route('/')
def main():
    return render_template("home.html")


@application.route('/selectModelYear', methods=['POST', 'GET'])
def selectModelYear():
    make = request.json['make']
    print(make)
    cursor = conn.cursor()
    query = "SELECT DISTINCT model_year FROM temp WHERE make = %s ORDER BY model_year ASC"
    cursor.execute(query, [make])
    result = cursor.fetchall()
    #return jsonify(tuple(i[0] for i in result))
    return json.dumps(result)

@application.route('/selectCarModel', methods=['POST', 'GET'])
def selectCarModel():
    make = request.json['make']
    print(make)
    cursor = conn.cursor()
    query = "SELECT DISTINCT model FROM temp WHERE make = %s ORDER BY model ASC"
    cursor.execute(query, [make])
    result = cursor.fetchall()

    #return jsonify(tuple(i[0] for i in result))
    return json.dumps(result)

@application.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        make = request.json['make']
        model = request.json['model']
        model_year = request.json['model_year']
        transmission = request.json['transmission']
        fuel_type = request.json['fuel_type']
        engine_capacity = request.json['engine_capacity']
        mileage = request.json['mileage']

    data = [[make, model, model_year, engine_capacity, mileage, 0, 0, 0, 0, 0, 0]]
    car_df = pd.DataFrame(data, columns=['make', 'model', 'model_year', 'engine_capacity', 'mileage',
                                         'transmission_Automatic', 'transmission_Manual',
                                         'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_Hybrid',
                                         'fuel_type_Petrol'])
    car_df = car_df.astype({"model_year": 'int64', "engine_capacity": 'int64', "mileage": 'int64'})

    car_df['make'] = car_df['make'].str.title()
    car_df['model'] = car_df['model'].str.title()
    print(car_df)

    make_col_enc = pickle.load(open('Data_Encoding/make_col_enc.pkl', 'rb'))
    car_df['make'] = make_col_enc.transform(car_df['make'])

    model_col_enc = pickle.load(open('Data_Encoding/model_col_enc.pkl', 'rb'))
    car_df['model'] = model_col_enc.transform(car_df['model'])

    if transmission == 'Automatic':
        car_df['transmission_Automatic'] = 1
    else:
        car_df['transmission_Manual'] = 1

    if fuel_type == 'Diesel':
        car_df['fuel_type_Diesel'] = 1
    elif fuel_type == 'Electric':
        car_df['fuel_type_Electric'] = 1
    elif fuel_type == 'Hybrid':
        car_df['fuel_type_Hybrid'] = 1
    else:
        car_df['fuel_type_Petrol'] = 1

    new_df = np.array(car_df)

    model = pickle.load(open('XGBoostModel.pkl','rb'))
    predicted_price = model.predict(new_df).tolist()

    print(predicted_price)
    return jsonify({'prediction_text': predicted_price})


if __name__ == '__main__':
    application.run(debug=True)

