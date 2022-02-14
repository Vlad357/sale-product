import pandas as pd
import numpy as np

path = 'Data/'
train = pd.read_csv(path + 'train.csv', parse_dates = ['date'], infer_datetime_format = True)
train = train.drop(columns = ['id']).copy()
test = pd.read_csv(path + 'test.csv', parse_dates = ['date'], infer_datetime_format = True).drop(columns = ['id'])
prediction_steps = test['date'].nunique()

stores = pd.read_csv(path + 'stores.csv').rename(columns = {'type': 'store_type', 'cluster': 'cluster_type'})

train = pd.merge(train, stores, on = 'store_nbr', how = 'left')

holidays = pd.read_csv(path + 'holidays_events.csv', parse_dates = ['date'], infer_datetime_format = True)
holidays = holidays.loc[holidays['transferred'] == False]

holidays_nat = holidays[holidays['locale']=='National'].drop_duplicates(subset='date')
holidays_reg = holidays[holidays['locale']=='Regional'].drop_duplicates(subset='date')
holidays_loc = holidays[holidays['locale']=='Local'].drop_duplicates(subset='date')

train = pd.merge(train, holidays_nat[['date', 'description']], on = 'date', how = 'left').rename(columns = {'description': 'holidays_nat'})
train = pd.merge(train, holidays_reg[['date', 'locale_name', 'description']], left_on = ['date', 'state'], right_on = ['date', 'locale_name'], how = 'left').rename(columns = {'description': 'holidays_reg'}).drop(columns = ['locale_name'])
train = pd.merge(train, holidays_loc[['date', 'locale_name', 'description']], left_on = ['date', 'city'], right_on = ['date', 'locale_name'], how = 'left').rename(columns = {'description': 'holidays_loc'}).drop(columns = ['locale_name'])
train[["holidays_nat", "holidays_reg", "holidays_loc"]] = train[["holidays_nat", "holidays_reg", "holidays_loc"]].fillna("No")

oil = pd.read_csv(path + 'oil.csv', parse_dates = ['date'], infer_datetime_format = True)

train = pd.merge(train, oil, on = 'date', how = 'left')

transactions = pd.read_csv(path + 'transactions.csv', parse_dates = ['date'], infer_datetime_format = True)
train = pd.merge(train, transactions, on = ['date', 'store_nbr'], how = 'left')

from scipy.stats import skewnorm
earthquake = pd.DataFrame()
earthquake['date'] = pd.date_range('2016-04-17', '2016-05-16')
earthquake['earthquake_effect'] = [2*skewnorm.pdf(i/20, 0.5) for i in range(len(earthquake))]

train = pd.merge(train, earthquake, on = 'date', how = 'left')
train['earthquake_effect'].fillna(0, inplace = True)

def get_distance_from_paydays(date):
    end_of_month = date.daysinmonth
    distance_to_1st = 0 if date.day >= 15 else 15 - date.day
    distance_to_15th = 0 if date.day < 15 else end_of_month - date.day
    return distance_to_1st + distance_to_15th

train['days_of_payday'] = train['date'].apply(get_distance_from_paydays)

train['average_sales_by_family'] = train.groupby(['date', 'family'], observed = True).sales.transform('mean')
train['average_sales_by_store'] = train.groupby(['date', 'store_nbr'], observed = True).sales.transform('mean')
train['dcoilwtico'] = train['dcoilwtico'].interpolate().fillna(method = 'bfill')

train['transactions'] = train['transactions'].interpolate().fillna(method = 'bfill')

train['dayofweek'] = train['date'].dt.dayofweek.astype('str').astype('category')
train['month'] = train['date'].dt.month.astype('str').astype('category')
train['deyofyear'] = train['date'].dt.dayofyear.astype('str').astype('category')
train['time_idx'] = (train['date'].dt.date - train['date'].dt.date.min()).dt.days

objCols = train.loc[:, train.select_dtypes('object').columns].copy()

def frames_to_maps(data_frame):
    for col in data_frame:
        types = []
        map_ = {'No': 0, 0:0}
        for i in col:
            if i not in types and i != 0:
                types.append(i)
        for i in range(len(types)):
            map_[types[i]] = i + 1
        yield map_


for i in range(len(objCols.columns)):
    objCols[objCols.columns[i]] = objCols[objCols.columns[i]].map(
        [x for x in frames_to_maps([objCols[x] for x in objCols.columns])][i]
    )

train.loc[:, objCols.columns] = objCols.loc[:, objCols.columns]  
train.to_csv('data_for_forecast.csv')