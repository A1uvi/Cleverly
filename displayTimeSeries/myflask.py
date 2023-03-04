from flask import Flask, render_template

import numpy as np
import pandas as pd
from fbprophet import Prophet
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import mpld3
from fbprophet_trying import create_dataset_for_prophet, get_model_params, setup_prophet_model, forecast_by_prophet_model, mean_absolute_percentage_error

from pathlib import Path

df = pd.read_csv('data_ops\electricity_data.csv')
plt.close('all')

nation = {"United_States": ["all_sectors","electric_utility","independent_power_producers"]}
regions = {"New_England":["all_sectors","electric_utility","independent_power_producers"],
           "Middle_Atlantic":["all_sectors", "electric_utility", "independent_power_producers","all_commercial"],
           "East_North_Central": ["all_sectors", "electric_utility", "independent_power_producers", "all_commercial"] ,
           "West_North_Central": ["all_sectors", "electric_utility","all_commercial"] ,
           "South_Atlantic":["all_sectors", "electric_utility", "independent_power_producers"],
           "East_South_Central": ["all_sectors", "electric_utility", "all_industrial"] ,
           "West_South_Central": ["all_sectors", "electric_utility", "independent_power_producers"] ,
           "Mountain": ["all_sectors","electric_utility"],
           "Pacific_Contiguous": ["all_sectors","electric_utility","independent_power_producers"]
           }
states = {"Connecticut": ["all_sectors","independent_power_producers"],
          "Maine": ["all_sectors"],
          "Massachusetts": ["all_sectors","independent_power_producers"],
          "New_Hampshire": ["all_sectors","independent_power_producers"],
          "Rhode_Island": ["all_sectors","independent_power_producers"],
          "Vermont": ["all_sectors","electric_utility"],
          "New_Jersey": ["all_sectors","independent_power_producers"],
          "New_York": ["all_sectors","electric_utility","independent_power_producers"],
          "Pennsylvania": ["all_sectors","independent_power_producers", "all_industrial"] ,
          "Illinois": ["all_sectors", "electric_utility", "independent_power_producers"] ,
          "Indiana": ["all_sectors","electric_utility","independent_power_producers"] ,
          "Michigan": ["all_sectors","electric_utility"] ,
          "Ohio": ["all_sectors","independent_power_producers"] ,
          "Wisconsin": ["all_sectors","electric_utility"] ,
          "Iowa": ["all_sectors"] ,
          "Kansas": ["all_sectors","electric_utility"] ,
          "Minnesota": ["all_sectors","electric_utility"] ,
          "Missouri": ["all_sectors","electric_utility"] ,
          "Nebraska": ["all_sectors","electric_utility"] ,
          "North_Dakota": ["all_sectors","electric_utility"] ,
          "South_Dakota": ["all_sectors"], 
          "Delaware": ["all_sectors"], 
          "Florida": ["all_sectors","electric_utility"] ,
          "Georgia": ["all_sectors", "electric_utility", "independent_power_producers"] ,
          "Maryland": ["all_sectors", "independent_power_producers"] ,
          "North_Carolina": ["all_sectors","electric_utility"] ,
          "South_Carolina": ["all_sectors", "electric_utility", "independent_power_producers"] ,
          "Virginia": ["all_sectors","electric_utility"] ,
          "West_Virginia": ["all_sectors","electric_utility"] ,
          "Alabama": ["all_sectors"], 
          "Kentucky": ["all_sectors","electric_utility"] ,
          "Mississippi": ["all_sectors","electric_utility", "all_industrial"] ,
          "Tennessee": ["all_sectors","electric_utility", "all_industrial"] ,
          "Arkansas": ["all_sectors"], 
          "Louisiana": ["all_sectors", "electric_utility", "independent_power_producers"] ,
          "Oklahoma": ["all_sectors","electric_utility", "all_industrial"] ,
          "Texas": ["all_sectors", "electric_utility", "independent_power_producers"] ,
          "Arizona": ["all_sectors","electric_utility"] ,
          "Colorado": ["all_sectors"], 
          "Idaho": ["all_sectors"], 
          "Montana": ["all_sectors","electric_utility"] ,
          "Nevada": ["all_sectors"], 
          "New_Mexico": ["all_sectors"], 
          "Utah": ["all_sectors"], 
          "Wyoming": ["all_sectors","electric_utility"] ,
          "California": ["all_sectors", "independent_power_producers"], 
          "Oregon": ["all_sectors","electric_utility"] ,
          "Washington": ["all_sectors", "independent_power_producers"], 
          "Alaska": ["all_sectors","electric_utility"] ,
          "Hawaii": ["all_sectors","electric_utility"]
          }

df_train = df[df['date'] < '2021-01-01']
df_valid = df[df['date'] >= '2021-01-01']
df_train_valid = df[df['date'] < '2021-01-01']
df_test = df[df['date'] >= '2021-01-01']

date_range = df_train['date'].unique()
date_range_train_valid = df_train_valid['date'].unique()

def predColsDict(colsDictKey, colsList):
    predicted = []
    imgPaths = {"all_sectors": "displayTimeSeries/static/imgs/graphs/0.png",
                "electric_utility": "displayTimeSeries/static/imgs/graphs/1.png",
                "independent_power_producers": "displayTimeSeries/static/imgs/graphs/2.png",
                "all_commercial": "displayTimeSeries/static/imgs/graphs/3.png",
                "all_industrial": "displayTimeSeries/static/imgs/graphs/4.png"
                }
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

        ax.legend()
        #plt.show(fig)
        #mpld3.show(fig)
        #plotdict = mpld3.fig_to_dict(fig)
        #for i in plotdict:
        #    print(i)
        #mpld3.save_html(fig,"trying_graph/templates/a.html")
        plt.savefig(Path(imgPaths[i]))
        plt.close(fig)

        predicted.append(i)
    #return predicted
'''
trial = "United_States"
print(nation[trial])
predColsDict(trial, nation[trial])
'''

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('a.html')
  
  
if __name__ == "__main__":
    app.run(host="localhost", port=int("5000"))
