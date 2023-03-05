from flask import Flask, render_template

import numpy as np
import pandas as pd
from fbprophet import Prophet
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from fbprophet_trying import create_dataset_for_prophet, get_model_params, setup_prophet_model, forecast_by_prophet_model, mean_absolute_percentage_error

from pathlib import Path

df = pd.read_csv('data_ops\electricity_data.csv')
plt.close('all')

nations = {"United_States": [["all_sectors","electric_utility","independent_power_producers"], "steady"]}

regions = {"New_England":[["all_sectors","electric_utility","independent_power_producers"], "description"],
           "Middle_Atlantic":[["all_sectors", "electric_utility", "independent_power_producers","all_commercial"], "description"],
           "East_North_Central": [["all_sectors", "electric_utility", "independent_power_producers", "all_commercial"], "description"] ,
           "West_North_Central": [["all_sectors", "electric_utility","all_commercial"], "description"] ,
           "South_Atlantic":[["all_sectors", "electric_utility", "independent_power_producers"], "description"],
           "East_South_Central": [["all_sectors", "electric_utility", "all_industrial"], "description"] ,
           "West_South_Central": [["all_sectors", "electric_utility", "independent_power_producers"], "description"] ,
           "Mountain": [["all_sectors","electric_utility"], "description"],
           "Pacific_Contiguous": [["all_sectors","electric_utility","independent_power_producers"], "description"]
           }

states = {"Connecticut": [["all_sectors","independent_power_producers"], "description"],
          "Maine": ["all_sectors"],
          "Massachusetts": [["all_sectors","independent_power_producers"], "description"],
          "New_Hampshire": [["all_sectors","independent_power_producers"], "description"],
          "Rhode_Island": [["all_sectors","independent_power_producers"], "description"],
          "Vermont": [["all_sectors","electric_utility"], "description"],
          "New_Jersey": [["all_sectors","independent_power_producers"], "description"],
          "New_York": [["all_sectors","electric_utility","independent_power_producers"], "description"],
          "Pennsylvania": [["all_sectors","independent_power_producers", "all_industrial"], "description"] ,
          "Illinois": [["all_sectors", "electric_utility", "independent_power_producers"], "description"] ,
          "Indiana": [["all_sectors","electric_utility","independent_power_producers"], "description"] ,
          "Michigan": [["all_sectors","electric_utility"], "description"] ,
          "Ohio": [["all_sectors","independent_power_producers"], "description"] ,
          "Wisconsin": [["all_sectors","electric_utility"], "description"] ,
          "Iowa": [["all_sectors"], "description"] ,
          "Kansas": [["all_sectors","electric_utility"], "description"] ,
          "Minnesota": [["all_sectors","electric_utility"], "description"] ,
          "Missouri": [["all_sectors","electric_utility"], "description"] ,
          "Nebraska": [["all_sectors","electric_utility"], "description"] ,
          "North_Dakota": [["all_sectors","electric_utility"], "description"] ,
          "South_Dakota": [["all_sectors"], "description"], 
          "Delaware": [["all_sectors"], "description"], 
          "Florida": [["all_sectors","electric_utility"], "description"] ,
          "Georgia": [["all_sectors", "electric_utility", "independent_power_producers"], "description"] ,
          "Maryland": [["all_sectors", "independent_power_producers"], "description"] ,
          "North_Carolina": [["all_sectors","electric_utility"], "description"] ,
          "South_Carolina": [["all_sectors", "electric_utility", "independent_power_producers"], "description"] ,
          "Virginia": [["all_sectors","electric_utility"], "description"] ,
          "West_Virginia": [["all_sectors","electric_utility"], "description"] ,
          "Alabama": [["all_sectors"], "description"], 
          "Kentucky": [["all_sectors","electric_utility"], "description"] ,
          "Mississippi": [["all_sectors","electric_utility", "all_industrial"], "description"] ,
          "Tennessee": [["all_sectors","electric_utility", "all_industrial"], "description"] ,
          "Arkansas": [["all_sectors"], "description"], 
          "Louisiana": [["all_sectors", "electric_utility", "independent_power_producers"], "description"] ,
          "Oklahoma": [["all_sectors","electric_utility", "all_industrial"], "description"] ,
          "Texas": [["all_sectors", "electric_utility", "independent_power_producers"], "description"] ,
          "Arizona": [["all_sectors","electric_utility"], "description"] ,
          "Colorado": [["all_sectors"], "description"], 
          "Idaho": [["all_sectors"], "description"], 
          "Montana": [["all_sectors","electric_utility"], "description"] ,
          "Nevada": [["all_sectors"], "description"], 
          "New_Mexico": [["all_sectors"], "description"], 
          "Utah": [["all_sectors"], "description"], 
          "Wyoming": [["all_sectors","electric_utility"], "description"] ,
          "California": [["all_sectors", "independent_power_producers"], "description"], 
          "Oregon": [["all_sectors","electric_utility"], "description"] ,
          "Washington": [["all_sectors", "independent_power_producers"], "description"], 
          "Alaska": [["all_sectors","electric_utility"], "description"] ,
          "Hawaii": [["all_sectors","electric_utility"], "description"]
          }

df_train = df[df['date'] < '2021-01-01']
df_valid = df[df['date'] >= '2021-01-01']
df_train_valid = df[df['date'] < '2021-01-01']
df_test = df[df['date'] >= '2021-01-01']

date_range = df_train['date'].unique()
date_range_train_valid = df_train_valid['date'].unique()

def predColsDict(colsDictKey, colsList):
    predicted = []
    for i in colsList:
        col_date = 'date'
        col_target = colsDictKey+"_"+i
        df[col_date] = pd.to_datetime(df[col_date])

        df_for_prophet = create_dataset_for_prophet(df_train, col_date=col_date, col_target=col_target)
        model = setup_prophet_model(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True, seasonality_mode='additive')

        df_forecast = forecast_by_prophet_model(df_for_prophet, model=model, steps_ahead=len(df_valid), freq='1M')
        df_forecast_pred_valid = pd.DataFrame({'date': df_valid[col_date].values, 'forecast': df_forecast['yhat'].values})
        df_valid_predicted_vs_actual = df_forecast_pred_valid.merge(df_valid, how='inner', on='date')

        fig, ax = plt.subplots()
        title = str(colsDictKey).replace("_"," ") + ": " + i.replace("_"," ").capitalize()
        ax.set_title(title)
        ax.plot(df_train['date'].iloc[-50:], df_train[col_target].iloc[-50:], label='Historical values')
        ax.plot(df_valid_predicted_vs_actual['date'], df_valid_predicted_vs_actual[col_target], label='Actual value')
        ax.plot(df_valid_predicted_vs_actual['date'], df_valid_predicted_vs_actual['forecast'], label='Forecast')

        myLocator = mticker.MultipleLocator(24)
        ax.xaxis.set_major_locator(myLocator)

        ax.legend()

        if i == "all_sectors":
            plt.savefig("displayTimeSeries/static/img/graphs/0.png")
        if i == "electric_utility":
            plt.savefig("displayTimeSeries/static/img/graphs/1.png")
        if i == "independent_power_producers":
            plt.savefig("displayTimeSeries/static/img/graphs/2.png")
        if i == "all_commercial":
            plt.savefig("displayTimeSeries/static/img/graphs/3.png")
        if i == "all_industrial":
            plt.savefig("displayTimeSeries/static/img/graphs/4.png")
        plt.close(fig)

        predicted.append(i)

descdict = {"steady": 'accurately',
            "description": "unknown"}
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/problem')
def problem():
    return render_template('problem.html')

@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/target')
def target():
    return render_template('target.html')

@app.route('/problemstatement')
def problemstatement():
    return render_template('problemstatement.html')

@app.route('/state/visoptions')
def visoptions():
    print(list(regions.keys()))
    return render_template('visoptions.html', regionNameList = list(regions.keys()))


@app.route('/nation')
def nation():
    trial = "United_States"
    print(nations[trial][0])
    predColsDict(trial, nations[trial][0])
    return render_template('showData.html', name = trial, pics = nations[trial][0], desc = nations[trial][1], result = descdict[nations[trial][1]], trial = trial.replace("_"," "), area = "National")


@app.route('/region/<regionname>')
def region(regionname):
    trial = regionname
    print(regions[trial][0])
    predColsDict(trial, regions[trial][0])
    return render_template('showData.html', name = trial, pics = regions[trial][0], desc = regions[trial][1], result = descdict[regions[trial][1]], trial = trial.replace("_"," "), area = "Regional")

@app.route('/state/<statename>')
def state(statename):
    trial = str(statename)
    print(states[trial][0])
    predColsDict(trial, states[trial][0])
    return render_template('showData.html', name = trial, pics = states[trial][0], desc = states[trial][1], result = descdict[states[trial][1]], trial = trial.replace("_"," "), area = "Individual State")

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")
  
if __name__ == "__main__":
    app.run(host="localhost", port=int("5000"))
