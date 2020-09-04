# -*- coding: utf-8 -*-
"""

@author: Zion
"""

#pull stock data
import pandas as pd
import random as rd
from glob import glob #for importing multiple csv's of same parent path
from numpy import log #default base is e (natural log)
import os.path
from os import path #to check valid file paths
import datetime
import json #to manipulate yec output
import pandas as pd
from yahoo_earnings_calendar import YahooEarningsCalendar
import numpy as np


def group_quarters(timelist):
    '''
    input: list of datetime objects from dataframe
    
    output: rounded list of datetime objects snapped to fixed quarter dates
    
    fixed dates (make sure not weekends, holidays)
    '''
#    qdates = ['0331', '0630', '0930', '1231']
    qdates = [331, 630, 930, 1231] #can't have 0 in front of month
#    formatted = [0]*4
#    for i in range(4):
#        formatted[i] = pd.datetime.strptime(qdates[i],'%m%d')
#    q1, q2, q3, q4 =  formatted
    q1, q2, q3, q4 =  qdates
    
    new_ind = [] #replacement quarter-based index
    for date in timelist:
        qlab = ''
        numdate = int(pd.datetime.strftime(date, '%m%d'))
        year = date.year
        if q1 <= numdate <= q2:
            qlab = 'Q1'
        elif q2 <= numdate <= q3:
            qlab = 'Q2'
        elif q3 <= numdate <= q4:
            qlab = 'Q3'
        elif q4 <= numdate or numdate <= q1: 
            qlab = 'Q4'
        new_ind.append(f'{year} {qlab}')
    return new_ind



#0) Pull list of SP500 tickers
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp5 = table[0]
tickers = list(sp5['Symbol'])
rd.seed(30) #use seed for reproducable experiment
stocks = rd.sample(tickers, 100)

stable_ticks = set() #keep track of tickers that exist in stooq set

#finding the path
#parent_path = 'C:/Users/Zion/Desktop/CorroTrack/stooq_d_us_txt/data/daily/us/nasdaq stocks/1/' #don't need full local path
parent_path = 'stooq_d_us_txt/data/daily/us/nasdaq stocks/' #all sp500 should be on the nasdaq
alt_path = 'stooq_d_us_txt/data/daily/us/nasdaq stocks/'
def fill_stocks():
    count = 0
    table = pd.DataFrame()
    for stock in stocks:
        if path.isfile(f'{parent_path}1/{stock}.us.txt'): #if valid file path
            stock_df = pd.read_csv(f'{parent_path}1/{stock}.us.txt')
            stable_ticks.add(stock)
        elif path.isfile(f'{parent_path}2/{stock}.us.txt'): #might be in other folder
            stock_df = pd.read_csv(f'{parent_path}2/{stock}.us.txt')
            stable_ticks.add(stock)
        else: #stooq doesnt contain data
            continue
        
        stock_df.drop(['<PER>', '<TIME>', '<VOL>', '<OPENINT>'], axis=1, inplace = True)

        #computing calculated values
        stock_df['Volatility'] = (0.361*(log(stock_df['<HIGH>']) - log(stock_df['<LOW>']))**2)**(0.5)
        stock_df['Return'] = (stock_df['<CLOSE>'] - stock_df['<OPEN>'])/stock_df['<OPEN>']*100

    
        times = pd.to_datetime(stock_df['<DATE>'], format ='%Y%m%d')
        
        #3) use ticker name to change the remaining columns
        c = stock_df.pivot(index='<TICKER>', columns = '<DATE>')
        
        #rename the columns somehow (like "AUSTRIA_Liquid Fuel")
        new = c.stack(0).transpose()
        new.columns = ['__'.join(col).strip() for col in new.columns.values]
        new.index = times.dt.strftime("%Y-%m-%d") #need to do the year first so index sorts correctly
        new = new[new.index >= "2015-01-01"] #crop this 
        if table.empty:
            table = new
        else: #ready to merge
            table=pd.merge(table, new, how='outer', on='<DATE>') #CHANGE IT TO MERGE ON OUTER BUT GO 5 YRS BACK
        count += 1
        print(f'Progress: {count} / {len(stocks)}')

    #Quarter condensing
    dtimes = pd.to_datetime(table.index, format ="%Y-%m-%d")
    table.index = group_quarters(dtimes) #need to cast back to a datetime object
    table.index.name = 'quarter'
    table = table.groupby(by="quarter").mean()
    table.sort_index(inplace = True)
    table = table.reindex(sorted(table.columns), axis=1)

    #filter out desired vars
    close_prices = table.filter(like='<CLOSE>', axis=1)
    volatility = table.filter(like='Volatility', axis=1)

    return table, close_prices, volatility

yec = YahooEarningsCalendar()

def pull_earnings(dtype = 'epssurprisepct'):
    earn_table= pd.DataFrame()
    prog = 0
    for stock in stable_ticks: #only pull earnings for valid stooq companies
        #only have to save the df once
        try:
            edf = pd.read_json(f'earnings/{stock}.json') #earnings data frame
        except: #dataframe doesn't exist yet    
            earnings = yec.get_earnings_of(stock)

            def save_json(data, stock): #pass in stock name and json    
                with open(f'earnings/{stock}.json', 'w') as json_file:
                    json.dump(data, json_file)

            save_json(earnings, stock)

            edf = pd.read_json(f'earnings/{stock}.json') #earnings data frame

        cropped_edf = edf[[dtype,'startdatetime', 'ticker']]

        #3) use ticker name to change the remaining columns
        #pivot_table = generalization of pivot that can aggregate duplicates
        c = cropped_edf.pivot_table(index='ticker', columns = 'startdatetime', aggfunc='mean')

        #rename the columns somehow (like "AUSTRIA_Liquid Fuel")
        new = c.stack(0).transpose()
        new.columns = ['__'.join(col).strip() for col in new.columns.values]
        dates = pd.to_datetime(new.index, format ='%Y-%m-%dT%H:%M:%S.%fZ')
        new.index = group_quarters(dates)
        new = new[new.index >= "2015 Q1"] #crop this 
        print(new.tail())

        if earn_table.empty:
            print('NO MERGE')
            earn_table = new
        else: #ready to merge
            print('MERGE')
            earn_table=pd.merge(earn_table, new, how='outer', left_index=True, right_index=True)#, on='startdatetime') #CHANGE IT TO MERGE ON OUTER BUT GO 5 YRS BACK
        prog += 1
        print(f'Progress: {prog} / {len(stable_ticks)}')

    earn_table.index.name = 'quarter'
    earn_table = earn_table.groupby(by="quarter").mean()
    earn_table = earn_table.reindex(sorted(earn_table.columns), axis=1)

    print(earn_table.tail(50))
    return earn_table

