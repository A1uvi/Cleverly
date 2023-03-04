
import pandas as pd
import numpy as np
from fbprophet import Prophet
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import mpld3

def create_dataset_for_prophet(df: pd.DataFrame, col_date: str, col_target: str, min_date = None, max_date = None) -> pd.DataFrame:
    '''
    Creating dataset ready to use with Prophet model. 
    It must have specific format: 
      - 'ds' column - a column containing dates
      - 'y' column - a column containing target variable values.
    There should be nothing except these two columns. 
    '''
    if min_date == None:
        
        min_date = df[col_date].min()
        
    if max_date == None:
        
        max_date = df[col_date].max()
    
    df_at_date_range = df[(df[col_date] >= min_date) & (df[col_date] <= max_date)]
    
    df_for_prophet = pd.DataFrame({'ds': df_at_date_range[col_date].values, 'y': df_at_date_range[col_target].values})
    
    return df_for_prophet

def get_model_params(**kwargs) -> Dict: 
    
    '''
    Helper function extracting Prophet model hyperparams from its keyword arguments.
    '''
    
    param_names = list(kwargs.keys())
        
    model_param_names = ['daily_seasonality', 'weekly_seasonality', 'yearly_seasonality', 'seasonality_mode']
        
    model_params = {}
    fit_params = {}
        
    for param_name in model_param_names:
            
        if param_name not in param_names:
                                    
            model_params[param_name] = None
            
        else:
                
            model_params[param_name] = kwargs[param_name]
                    
    return model_params
  
def setup_prophet_model(**kwargs) -> Prophet:
    
    '''
    Setting up hyperparams for the Prophet model. 
    For now, they include presence of daily/weekly/yearly seasonalities and additive/multiplicative mode of seasonality.
    '''
    
    model_params = get_model_params(**kwargs)
    model = Prophet(**model_params)
    
    return model

def forecast_by_prophet_model(df_for_prophet: pd.DataFrame, model: Prophet, steps_ahead: int = 1, freq: str = 'D') -> pd.DataFrame:
    
    '''
    Creating a forecast using Prophet model and a dataset prepared in advance.
    '''
    
    # In case the model is not fit yet, we fit it. Otherwise, we pass this step and go ahead with an already-fit model as it can be fit only once.
    try:    
    
        model.fit(df_for_prophet)
    
    except Exception:
        
        pass
    
    df_future = model.make_future_dataframe(periods=steps_ahead, freq=freq)
    df_predict = model.predict(df_future).iloc[-steps_ahead:]
    
    return df_predict
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics(y_true, y_pred):
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f'MAE = {mae:.2f}')
    print(f'RMSE = {rmse:.2f}')
    print(f'MAPE = {mape:.2f}%')

df = pd.read_csv('data_ops\electricity_data.csv')


listCols = list(df.columns)[1:]
invalid = ['Rhode_Island_electric_utility', 'Rhode_Island_all_commercial', 'Rhode_Island_all_industrial', 'Vermont_all_commercial', 'Vermont_all_industrial', 'Ohio_all_commercial', 'Kansas_independent_power_producers', 'Kansas_all_commercial', 'Kansas_all_industrial', 'Missouri_independent_power_producers', 'North_Dakota_independent_power_producers', 'North_Dakota_all_commercial', 'South_Dakota_independent_power_producers', 'South_Dakota_all_commercial', 'South_Dakota_all_industrial', 'Delaware_all_commercial', 'District_Of_Columbia_electric_utility', 'District_Of_Columbia_independent_power_producers', 'District_Of_Columbia_all_commercial', 'District_Of_Columbia_all_industrial', 'Maryland_electric_utility', 'West_Virginia_all_commercial', 'Alabama_all_commercial', 'Kentucky_all_commercial', 'Oklahoma_all_commercial', 'Arizona_all_industrial', 'Colorado_all_commercial', 'Idaho_all_commercial', 'Montana_all_commercial', 'Nevada_all_commercial', 'Nevada_all_industrial', 'Utah_all_industrial', 'Wyoming_all_commercial', 'Hawaii_all_commercial']
for i in invalid:
    if i in listCols:
        listCols.remove(i)
#print(listCols)

df_train = df[df['date'] < '2021-01-01']
df_valid = df[df['date'] >= '2021-01-01']
df_train_valid = df[df['date'] < '2021-01-01']
df_test = df[df['date'] >= '2021-01-01']

date_range = df_train['date'].unique()
date_range_train_valid = df_train_valid['date'].unique()

invalid = []

'''
for i in listCols[0:3]:
    col_date = 'date'
    col_target = i
    df[col_date] = pd.to_datetime(df[col_date])

    df_for_prophet = create_dataset_for_prophet(df_train, col_date=col_date, col_target=col_target)
    model = setup_prophet_model(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True, seasonality_mode='additive')

    try:
        df_forecast = forecast_by_prophet_model(df_for_prophet, model=model, steps_ahead=len(df_valid), freq='1M')
        df_forecast_pred_valid = pd.DataFrame({'date': df_valid[col_date].values, 'forecast': df_forecast['yhat'].values})
        df_valid_predicted_vs_actual = df_forecast_pred_valid.merge(df_valid, how='inner', on='date')

        fig, ax = plt.subplots()
        ax.set_title(str(col_target))
        ax.plot(df_train['date'].iloc[-50:], df_train[col_target].iloc[-50:], label='Historical values')
        ax.plot(df_valid_predicted_vs_actual['date'], df_valid_predicted_vs_actual[col_target], label='Actual value')
        ax.plot(df_valid_predicted_vs_actual['date'], df_valid_predicted_vs_actual['forecast'], label='Forecast')

        ax.legend()
        plt.show(fig)

        #mpld3.show(fig)
        #plotdict = mpld3.fig_to_dict(fig)
        #for i in plotdict:
        #    print(i)
        #mpld3.save_html(fig,"trying_graph/templates/a.html")
    except:
        invalid.append(i)
        '''

'''
col_date = 'date'
col_target = "Maryland_electric_utility"
df[col_date] = pd.to_datetime(df[col_date])

df_for_prophet = create_dataset_for_prophet(df_train, col_date=col_date, col_target=col_target)
model = setup_prophet_model(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True, seasonality_mode='additive')

df_forecast = forecast_by_prophet_model(df_for_prophet, model=model, steps_ahead=len(df_valid), freq='1M')
df_forecast_pred_valid = pd.DataFrame({'date': df_valid[col_date].values, 'forecast': df_forecast['yhat'].values})
df_valid_predicted_vs_actual = df_forecast_pred_valid.merge(df_valid, how='inner', on='date')

fig, ax = plt.subplots()
ax.set_title(str(col_target))
ax.plot(df_train['date'].iloc[-50:], df_train[col_target].iloc[-50:], label='Historical values')
ax.plot(df_valid_predicted_vs_actual['date'], df_valid_predicted_vs_actual[col_target], label='Actual value')
ax.plot(df_valid_predicted_vs_actual['date'], df_valid_predicted_vs_actual['forecast'], label='Forecast')

ax.legend()
plt.show(fig)'''

#mpld3.show(fig)
#plotdict = mpld3.fig_to_dict(fig)
#for i in plotdict:
#    print(i)
#mpld3.save_html(fig,"trying_graph/templates/a.html")

print("DONE")