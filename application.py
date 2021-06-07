
import yfinance as yf
import datetime as dt

import pandas as pd
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,LassoLars,Ridge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split


import requests
from pandas_datareader import data
from flask import Flask, request, render_template, session, redirect

import time
import json

application=Flask(__name__)



# pip list --format=freeze >requirements.txt
model,accuracy,bs_error,squared_error,rms_error=0,0,0,0,0
savemetric=[]
stockdata=pd.DataFrame()       

_EXCHANGE_LIST = ['nyse', 'nasdaq', 'amex']

headers = {
    'authority': 'api.nasdaq.com',
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
    'origin': 'https://www.nasdaq.com',
    'sec-fetch-site': 'same-site',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://www.nasdaq.com/',
    'accept-language': 'en-US,en;q=0.9',
}

_SECTORS_LIST = set(['Consumer Non-Durables', 'Capital Goods', 'Health Care',
                     'Energy', 'Technology', 'Basic Industries', 'Finance',
                     'Consumer Services', 'Public Utilities', 'Miscellaneous',
                     'Consumer Durables', 'Transportation'])

def params(exchange):
    return (
        ('letter', '0'),
        ('exchange', exchange),
        ('render', 'download'),
    )

params = (
    ('tableonly', 'true'),
    ('limit', '25'),
    ('offset', '0'),
    ('download', 'true'),
)

@application.route('/')
def index():
    
    a=5
    b=100
    #return "Hello,Beautiful"
    sum=add(a,b)
    df=stock_model()
    #dataf=pd.read_csv("static/Dataset.csv")
    #dataf=dataf.to_json()
    #str1=json.loads(dataf)
    return render_template("index.html",data=df)
    # return str(a+b)
    # return sum
    # return df.to_html(header="true", table_id="table")
   

def add(a,b):
    return str(a+b)
    
def get_tickers_filtered(mktcap_min=None, mktcap_max=None, sectors=None):
    tickers_list = []
    for exchange in _EXCHANGE_LIST:
        tickers_list.extend(
            __exchange2list_filtered(exchange, mktcap_min=mktcap_min, mktcap_max=mktcap_max, sectors=sectors))
    return tickers_list                                       

def __exchange2df(exchange):
    # response = requests.get('https://old.nasdaq.com/screening/companies-by-name.aspx', headers=headers, params=params(exchange))
    # data = io.StringIO(response.text)
    # df = pd.read_csv(data, sep=",")
    r = requests.get('https://api.nasdaq.com/api/screener/stocks', headers=headers, params=params)
    data = r.json()['data']
    df = pd.DataFrame(data['rows'], columns=data['headers'])
    return df
                          
def __exchange2list_filtered(exchange, mktcap_min=None, mktcap_max=None, sectors=None):
    df = __exchange2df(exchange)
    # df = df.dropna(subset={'MarketCap'})
    df = df.dropna(subset={'marketCap'})
    df = df[~df['symbol'].str.contains("\.|\^")]

    if sectors is not None:
        if isinstance(sectors, str):
            sectors = [sectors]
        if not _SECTORS_LIST.issuperset(set(sectors)):
            raise ValueError('Some sectors included are invalid')
        sector_filter = df['sector'].apply(lambda x: x in sectors)
        df = df[sector_filter]

    def cust_filter(mkt_cap):
        if 'M' in mkt_cap:
            return float(mkt_cap[1:-1])
        elif 'B' in mkt_cap:
            return float(mkt_cap[1:-1]) * 1000
        elif mkt_cap == '':
            return 0.0
        else:
            return float(mkt_cap[1:]) / 1e6

    df['marketCap'] = df['marketCap'].apply(cust_filter)
    if mktcap_min is not None:
        df = df[df['marketCap'] > mktcap_min]
    if mktcap_max is not None:
        df = df[df['marketCap'] < mktcap_max]
    return df['symbol'].tolist()
                 
def store_metric(metrics_df):
    global metric_frames
    metric_frames.append(metrics_df)
    
def get_stock_data(symbol, from_date, to_date):
    data = yf.download(symbol, start=from_date, end=to_date)
    df = pd.DataFrame(data=data)
    global stockdata
    # stockdata['Dates']
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HighLoad', 'Change', 'Volume']]
    # if symbol=="WFC":
    #     print(df)
    return df

def stock_forecasting(df,modelname,stock_symbol):
    # if stock_symbol=="WFC":
        # print(df)
    forecast_col = 'Close'
    # forecast_out = int(math.ceil(0.01*len(df)))
    forecast_out =1

    # print("Value=",forecast_out)
    df['Label'] = df[[forecast_col]].shift(-forecast_out)
    # df['Label'] = df[[forecast_col]]

    X = np.array(df.drop(['Label'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)
    y = np.array(df['Label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if modelname=='Stepwise':
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    elif modelname=='Svm':
        clf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    elif modelname=='Lasso':
        clf =LassoLars(alpha=.1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
    
    elif modelname=='Ridge':
        clf =Ridge(alpha=.1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    elif modelname=='Boosted':
        clf =GradientBoostingRegressor(random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    else:
        clf = RandomForestRegressor(n_estimators=20, random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
    
    forecast = clf.predict(X_forecast)
    metrics_df=[]
    test_predict=clf.predict(X_test)

    bs_error=metrics.mean_absolute_error(y_test,test_predict ) 
    squared_error=metrics.mean_squared_error(y_test,test_predict)
    rms_error=np.sqrt(metrics.mean_squared_error(y_test,test_predict))
    
    metrics_df.append(model)
    metrics_df.append(stock_symbol)
    metrics_df.append(accuracy)
    metrics_df.append(bs_error)
    metrics_df.append(squared_error) 
    metrics_df.append(rms_error)
   
    df['Prediction'] = np.nan
    last_date = df.iloc[-1].name
    # print(last_date)
    last_date = dt.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S")
    pred_arr=[]
    date_field=[]
    for pred in forecast:
        pred_arr.append(pred)
        last_date += dt.timedelta(days=1)
        df.loc[last_date.strftime("%Y-%m-%d")] = [np.nan for _ in range(len(df.columns) - 1)] + [pred]
        # df.loc[last_date.strftime("%Y-%m-%d")] = [date_field.append(last_date.strftime("%Y-%m-%d")) for _ in range(len(df.columns) - 1)] + [pred]
        date_field.append(last_date.strftime("%Y-%m-%d"))

    return df, forecast_out,pred_arr,date_field,metrics_df
#Evaluation Metric generation
# change the model name to generate the specific model's evaluation metric
def metric_calculation(metrics_df,count,total_tickers,model_name):
    global model,accuracy,bs_error,squared_error,rms_error
    global savemetric
    model=metrics_df[0]
    if count<total_tickers:
        accuracy+=metrics_df[2]   
        bs_error+=metrics_df[3] 
        squared_error+=metrics_df[4]
        rms_error+=metrics_df[5]
        
    elif count==total_tickers:
        accuracy+=metrics_df[2]   
        bs_error+=metrics_df[3] 
        squared_error+=metrics_df[4]
        rms_error+=metrics_df[5]
        
        accuracy=round(accuracy/total_tickers,4)
        bs_error=round(bs_error/total_tickers,4)
        rms_error=round(rms_error/total_tickers,4)
        squared_error=round(squared_error/total_tickers,4)
        
        savemetric.append(model)
        savemetric.append(accuracy)
        savemetric.append(bs_error)
        savemetric.append(squared_error)
        savemetric.append(rms_error)
        df=pd.DataFrame(savemetric)
        df_out=df.copy() 
        model,accuracy,bs_error,squared_error,rms_error =0,0,0,0,0
        savemetric=[]
        return df_out
   
def choose_rf(dataframe,rf):
    
    final_model=""
    dataframe_copy=dataframe.copy()
    list_of_models=dataframe.iloc[[0]].values.tolist()
    dataframe_copy.columns=list_of_models    
    accuracy=dataframe_copy.iloc[[1]]
    rms_error=dataframe_copy.iloc[[4]]
    dataframe_copy.drop(0,axis=0,inplace=True)
    # dataframe_copy= dataframe_copy.sort_values(dataframe_copy.columns[1],ascending=False)
    if (rf=="0.01"):
        
        #get the model from the indexes 
        accuracy=accuracy.set_index(pd.Index(["Accuracy"]))
        accuracy=accuracy.sort_values(by="Accuracy",ascending=False,axis=1)
        model_to_choose=accuracy.iloc[0].index.values
        final_model=str(model_to_choose[0])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

    if (rf=="0.02"):
        
        #get the model from the indexes 
        accuracy=accuracy.set_index(pd.Index(["Accuracy"]))
        accuracy=accuracy.sort_values(by="Accuracy",ascending=False,axis=1)
        model_to_choose=accuracy.iloc[0].index.values
        final_model=str(model_to_choose[1])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

    
    if (rf=="0.03"):
        
        #get the model from the indexes 
        rms_error=rms_error.set_index(pd.Index(["RMS"]))
        rms_error=rms_error.sort_values(by="RMS",ascending=False,axis=1)
        model_to_choose=rms_error.iloc[0].index.values
        final_model=str(model_to_choose[0])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

    if (rf=="0.04"):
        
        #get the model from the indexes 
        rms_error=rms_error.set_index(pd.Index(["RMS"]))
        rms_error=rms_error.sort_values(by="RMS",ascending=False,axis=1)
        model_to_choose=rms_error.iloc[0].index.values
        final_model=str(model_to_choose[1])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

        
    return final_model

def stock_model():
        # models=[]
        # evaluation_metric=np.array([])
        # actual_date = dt.date.today()
        # historic_date=pd.DataFrame()
        # eval_metric=[]
        # stock_size_counter=[]
        # past_date = actual_date - dt.timedelta(days=365 * 3)
        # # symbol="AAPL"
        # actual_date = actual_date.strftime("%Y-%m-%d")
        # past_date = past_date.strftime("%Y-%m-%d")
        # filtered_tickers = get_tickers_filtered(mktcap_min=80000, mktcap_max=10000000)
        # filtered_tickers=list(dict.fromkeys(filtered_tickers))
        # total_tickers=len(filtered_tickers)
        # model_error=[]
        # add_metric=np.array([])
        timelen=time.time()

        rf=0.02
        # Final_Metric=pd.read_csv('Final_Metric.csv')
        # Final_Metric.drop(columns=['Unnamed: 0'],inplace=True)
        # final_stock_tickers=[]
        # out=choose_rf(Final_Metric,rf)
        # models.append(out)
        # for model in models:
        #     count=0
        #     model_name=model
        #     predicted_df=np.array([])
        #     historic_data=np.array([])
        #     for stock_symbol in filtered_tickers:
        #         count+=1
                
        #         if count<=total_tickers:
        #             prev_data=pd.DataFrame()
        #             # print(' {} {}/{} '.format(stock_symbol,count,total_tickers))    
        #             dataframe = get_stock_data(stock_symbol, past_date, actual_date)
        #             prev_data=dataframe.copy()
        #             prev_dates=dataframe.copy()
        #             # moving data index
        #             prev_dates=prev_dates.reset_index()
        #             prev_dates=prev_dates['Date']
        #             historic_date=prev_dates
                    
        #             prev_data=prev_data['Close'].values.tolist()
                    
        #             prev_data=np.array([prev_data]).T
        #             stock_size_counter.append(prev_data.size)
        #             if (len(stock_size_counter)==1 or stock_size_counter[-2]==stock_size_counter[-1]):
                        
        #                 historic_data=np.hstack([historic_data,prev_data]) if historic_data.size else prev_data
        #                 (dataframe, forecast_out,pred_arr,date_field,metrics_df) = stock_forecasting(dataframe,model,stock_symbol)
                        
        #                 df_out_metric=metric_calculation(metrics_df,count,total_tickers,model)
        #                 if(count==total_tickers):
        #                     add_metric=np.hstack([add_metric,np.array(df_out_metric.values.tolist())]) if add_metric.size else np.array(df_out_metric.values.tolist())   
        #                 pred_values=np.array([pred_arr]).T
                     
        #                 final_stock_tickers.append(stock_symbol)
        #                 metric_values=np.array([metrics_df]).T
        #                 evaluation_metric=np.hstack([evaluation_metric,metric_values]) if evaluation_metric.size else metric_values
        #                 predicted_df=np.hstack([predicted_df,pred_values]) if predicted_df.size else pred_values
                   
        #     final_dataframe=pd.DataFrame(predicted_df)
        #     historic_dataframe=pd.DataFrame(historic_data)
        #     historic_dataframe.columns=final_stock_tickers
        #     historic_dataframe.insert(0,"Date",historic_date.values.tolist())
        #     historic_dataframe['Date']=pd.to_datetime(historic_dataframe['Date'])
        #     historic_dataframe.to_csv("Historic_Data.csv")
        #     final_dataframe.columns=final_stock_tickers
        #     final_dataframe.insert(0,"Date",date_field)
        #     final_dataframe.to_csv("Dataset.csv") 
            
        # add_df=pd.DataFrame(add_metric)
        # add_df.to_csv("Final_Metric.csv")

        # print('Printing the data frame...')
        # print(data)
   
        df_h=pd.read_csv("static/Historic_Data.csv")
        df_h=df_h.drop(["Unnamed: 0"],axis=1)
        
        # df=historic_dataframe.copy()
        #Pass on to the recommendation function
        recommendation_array=recommendation_model(df_h, rf)
        print(time.time()-timelen)
        # return dataframe,risk_factor 
        return recommendation_array
       
def recommendation_model(df,rf):
        # df=historic_dataframe.copy()
        date_df=df.reset_index()
        date_df=df['Date'].values.tolist()

        stock=df.iloc[0,:]
        stock=stock.reset_index()
        stock=stock[['index']]
        stock=stock.drop([0,1]).values
        stock=stock.flatten()
        df=df.drop(df.columns[0],axis=1)
        df=df[stock] 

        cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
        corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
# df=pd.DataFrame(date_df,columns=stock,)
        df.insert(0,"Date",date_df)
        df['Date']=pd.to_datetime(df['Date'])
        df=df.set_index("Date")
#Calculate Yearly index 
        ind_er = df.resample('Y').last().pct_change().mean()
#Calculating volatility and Returns 
        ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

# print(ann_sd)
        assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
        assets.columns = ['Returns', 'Volatility']
# print(assets)
        p_ret = [] # Define an empty array for portfolio returns
        p_vol = [] # Define an empty array for portfolio volatility
        p_weights = [] # Define an empty array for asset weights
        num_assets = len(df.columns)
        num_portfolios = 1000
        time1=time.time()
        for portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights = weights/np.sum(weights)
            p_weights.append(weights)
            returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
            p_ret.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
            sd = np.sqrt(var) # Daily standard deviation
            ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
            p_vol.append(ann_sd)
        print(time.time()-time1)
        data = {'Returns':p_ret, 'Volatility':p_vol}

        for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
            data[symbol] = [w[counter] for w in p_weights]
        portfolios  = pd.DataFrame(data)
# print(portfolios.head())
        min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
        # rf = 0.01 # risk factor
        
        optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
        # print(optimal_risky_port)
        
        top_opr=optimal_risky_port.to_frame()
        returns=round(top_opr.iloc[0,0],3)*100
        volatility=round(top_opr.iloc[1,0],3)*100
        top_opr=top_opr.drop(['Returns','Volatility'])
        top_opr_copy=top_opr.copy()
        top_opr_copy=top_opr.reset_index()
        
        val=len(top_opr_copy)
        for i in range(0,val):
            top_opr_copy.iloc[i,1]= top_opr_copy.iloc[i,1]*100
            top_opr_copy.iloc[i,1]=round(top_opr_copy.iloc[i,1],3)
            i+=1

        top_opr_copy=top_opr_copy.sort_values(top_opr_copy.columns[1],ascending=False)

    
        return top_opr_copy
    
if __name__ == '__main__':


          # stock_model() 
        application.run(debug=True)
