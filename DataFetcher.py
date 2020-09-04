# -*- coding: utf-8 -*-

#import numpy
from datapackage import Package
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Constants:
    
    TOKENS = {
        'usda': '', #us dept. of agriculture, economic research service
        'eia': '', #us energy info administration
    } 
    
    URLS = {
            'emissions': 'https://datahub.io/core/co2-fossil-by-nation/datapackage.json',
            'rd': 'https://datahub.io/core/expenditure-on-research-and-development/datapackage.json',
            'refugee': 'https://datahub.io/world-bank/sm.pop.refg/datapackage.json',
            'fertilizer': 'https://datahub.io/world-bank/ag.con.fert.zs/datapackage.json'
            }


def url2df(url):
    '''
    converts url to pandas dataframe
    '''
    package = Package(url)
    resources = package.resources
    data = pd.read_csv(resources[-1].descriptor['path']) #seems like last resource always has main data
    if data.empty: #last resource wasn't populated
        for resource in resources:
            if resource.tabular:
                data = pd.read_csv(resource.descriptor['path'])
    return data

def format_df(df, tag = None):
    '''
    input: df with Columns 'Country' & 'Year'
         - tag (optional): add supplementary descriptor to end of col label
    
    format to following criteria:
        + columns of form country_data
        + year index
    '''
    #1) pivot & format df's 
    df= df.pivot(index='Country', columns = 'Year')
    
    df= pd.DataFrame(df.stack(0).transpose()) #condense columns into one
    #rename the columns somehow (like "AUSTRIA_Liquid Fuel")
    if tag:
        tag = ' ' + tag
    else:
        tag = ''
    df.columns = ['__'.join(col).strip()+ tag for col in df.columns.values]
    return df

#emissions data
def fetch_fossFuel():
    emissions_data = url2df(Constants.URLS['emissions'])
    #convert to proper format
    emissions = format_df(emissions_data, 'Emissions') 
    
    return emissions #good!

#international research & development budget data
def fetch_rd():
    rd_data = url2df(Constants.URLS['rd'])
    
    rd_data.rename(columns = {'TIME': 'Year'}, inplace = True)
    
    rd = format_df(rd_data.iloc[:,1:], 'R&D Fund')
    
    return rd

#Refugee population by country or territory of asylum
def fetch_refPop():
    refugee_data = url2df(Constants.URLS['refugee'])
    
    refugee_data.rename(columns = {'Country Name': 'Country',
                                   'Value': 'Refugee Pop'}, inplace = True)
    refugee_data.drop(columns = ['Country Code'], inplace=True) #take out country code column
    
    ref_pop = format_df(refugee_data)
    
    return ref_pop

# Fertilizer consumption (kilograms per hectare of arable land)
def fetch_fert():
    fertilizer_data = url2df(Constants.URLS['fertilizer'])
    
    fertilizer_data.rename(columns = {'Country Name': 'Country',
                                   'Value': 'Fertilizer Cons'}, inplace = True)
    fertilizer_data.drop(columns = ['Country Code'], inplace=True) #take out country code column
    
    fert_cons = format_df(fertilizer_data)
    
    return fert_cons

#Methods
#get all the data we have
def full_fetch():
    '''
    Returns all available datasets as a large merged panda df 
    '''
    #instantiation
    merged = fetch_fossFuel() 
    fetch_list = []
    
    #add all the functions (ongoing)
    fetch_list.append(fetch_rd)
    fetch_list.append(fetch_refPop)
    fetch_list.append(fetch_fert)
    
    for fetcher in fetch_list:
        merged = pd.merge(merged, fetcher(), how='inner', left_index = True, right_index=True)
        #try how='outer' to test Tyler's theory
        #change dropna in [most_correlated] from axis=1 to axis=0
 
    return merged


def comp_labels(lab1, lab2=None, df=pd.DataFrame()):
    '''
    input: 2 labels from existing data set. default label 2 is data most 
        correlated w/ label 1
        
        output: correlation coefficient, matplotlib chart of overlayed datasets 
    '''
    if df.empty: #set default dataframe to full_fetch data
        df = full_fetch()

    corrMat = df.dropna(axis=1).corr()
#    corrMat = df.corr()
    corr= most_correlated(lab1, 0, corrMat)

    if not lab2: #set lab2 to default
        lab2 = corr.index.tolist()[-1] #label of data most correlated w/lab1
        corrCoeff = corr[-1]
        print('Most Correlated Data: ', lab2, '\n')
    else:
        corrCoeff =corr.loc[lab2]
    
    print('Correlation Coefficient: ', corrCoeff)
    
    # Pull plot data
    t = df.index
    data1 = df[lab1]
    data2 = df[lab2]
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Years')
    ax1.set_ylabel(lab1, color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) #cast x-axis ticks to ints
#    ax1.set_xlim(2013, 2016)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(lab2, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

comp_labels('Poland__Refugee Pop')
full_df = full_fetch()    
    
#%%
#exploring time offset - 3 steps

def lagging_ind(label, offset=None, df = pd.DataFrame()): 
    '''
    Objective: Find lagging indicator of label 
    
    input: 
        label of interest
        time lag offset (years)- default: offset for most correlated lag. max val = 25% of data span
        dataframe- default: full_fetch df 
        
    output: correlation coefficient + plot of best lagging indicator
    '''
    #helper fcn to output time-offset df 
    def df_shift(dframe, tlag=1, tlab=None):
        if not tlab:
            tlab = tlag
        
        #extract, turn to datetime, add year, turn back to index
        years = dframe.index
        dt = pd.to_datetime(years, format = '%Y') #index of int -> datetime years
        #make new offset df
        dt=pd.to_datetime(dt.year + tlag, format='%Y')
        dframe.index = dt.year
        t = 'year' if tlab == 1 else 'years'
        tag = f' -- {tlab} {t} ahead'
        dframe.columns =[col + tag for col in dframe.columns.values] #rename for col
        return dframe #fcn MODIFIES whatever df is passed in. avoids expensive copy
        
    #0) extract data column
    x = df[label].dropna()
    
    #1) Duplicate original df.
    df2 = df.copy()
    
    #2) Shift df2 index into past/future. Relabel columns to reflect time shift
    if offset:
        window = 2 #experiment with different window sizes
        tol = .1 #tolerance to price in tradeoff of time offset
        df2 = df_shift(df2, offset)
        
        df_window_forward = df2.copy() #reference is shifted frame
        df_window_back = df2.copy()

        #3) Re-merge, re-calculate
        df3 = pd.merge(x, df2, how='inner', left_index = True, right_index=True)
        #give precedence to requested frame - if next best corr isn't better by
        #corr1 + tol, then stick with corr1
        corr1 = most_correlated(label, top=1, df = df3, weighted = True)[0]
        
        #COMPUTE TIME LAGGED W/ CROSS CORRELATION
        orig_labels = df.columns        
        #do forward & backward for loop
        for i in range(1, window+1):
            back_off = -i #backwards offset
            fwd = df_shift(df_window_forward, tlab = i) #i=offset. increment each iter
            rev = df_shift(df_window_back, -1, tlab = back_off)
            #need to reset shifted columns before merge
            df4 = pd.merge(df3, fwd, how='inner', left_index = True, right_index=True)
            df4 = pd.merge(df3, fwd, how='inner', left_index = True, right_index=True)
            fwd.columns = orig_labels
            rev.columns = orig_labels
        corr2 = most_correlated(label, top=1, df = df4, weighted = True)[0]
        
        if corr2 > corr1+tol: #window found better vals
            df3 = df4
    
    else: #IF NO OFFSET, calculate optimal offset
        span = x.index[-1] - x.index[0]
        df3 = x.copy()
        orig_labels = df2.columns #keep tracck of original cols for labelling
        #make a huge dataframe of all the time offsets  and use comp_labels to find best
        for i in range(1, span//4 + 1):
        
            try: #FOR DEBUGGING
                shifted = df_shift(df2, tlab = i) #i=offset. increment each iter
            except: #assume it's OutOfBoundsDatetime error
                print('DateTimeError: ', df2.index[-1],i)
            #need to reset shifted columns before merge
            print('DEBUG - MTTF: ', i, span//4)
            df3 = pd.merge(df3, shifted, how='inner', left_index = True, right_index=True)
            shifted.columns = orig_labels

    comp_labels(label, df=df3)
    

#%%
def most_correlated(label, top=0, df = pd.DataFrame(), weighted = False):
    '''
    input: valid column label from merged_df; dframe from which to compute corr Matrix
        - x: number of items in return list
        - weighted: if true, correlations weighted based on how many overlapping
            data values there are 
    
    output: series w/ list of all correlated rv's arranged in order of highest
    to lowest corr
    '''
    if df.empty: #there's no dataset to compare against
        df = full_fetch()
    corrMat = df[df.columns].apply(lambda x: x.corr(df[label]))
    print(corrMat)
    series = corrMat.round(2) #isolate column; round to hundreth
    
    if weighted: #need to factor in how much each col overlaps with rv in question
        tuner = 1 #tweak tuner to change how much matching time span matters
        #1) find the range in the df for which the label of interest is defined
        x = df.loc[:, label].dropna() # isolate col of interest
        #crop df based on rv of interest
        df = df.truncate(before=x.index[0], after = x.index[-1])
        denom = x.count()

        mult = df.count()/denom * tuner #list of multipliers
        #make a new col of weighted correlations, merge in that 
        wcol = series * mult
        weight_df = pd.merge(series.rename("Corr"), wcol.rename('Weighted Corr'), how='outer', left_index = True, right_index=True)
        #pruning
        weight_df.dropna(inplace=True)
        #sort from least to greatest. maybe good to sort by absolute vals
        weight_df.drop(labels = [lab for lab in weight_df.index.tolist() if label in lab], inplace = True) #don't need to see correlation with shifted versions of self        
        weight_df = weight_df[(weight_df.T != 0).any()] #filter out 0's
        weight_df.sort_values(by = 'Weighted Corr', inplace = True) #sort it by weighted vals of last col
        return weight_df.iloc[-top:, 0] #correlation series, sorted by weighted corr
        
    else:
        #pruning
        series.dropna(inplace=True)
        #sort from least to greatest. maybe good to sort by absolute vals
        series.drop(index=label, inplace=True) #don't need to see correlation with itself
        series = series[series != 0] #filter out 0's
        #sort by the last column 
        return series.sort_values().iloc[-top:] 

    

def comp_labels(lab1, lab2=None, df=pd.DataFrame()):
    '''
    input: 2 labels from existing data set. default label 2 is data most 
        correlated w/ label 1
        
        output: correlation coefficient, matplotlib chart of overlayed datasets 
    '''
    if df.empty: #set default dataframe to full_fetch data
        df = full_fetch()
    corr= most_correlated(lab1, 0, df, True)
    if not lab2: #set lab2 to default
#        lab2 = corr.index.tolist()[-1] #label of data most correlated w/lab1
        lab2 = corr.index.tolist()[-1] #label of data most correlated w/lab1
        corrCoeff = corr[-1]
        print('Most Correlated Data: ', lab2, '\n')
    else:
        corrCoeff =corr.loc[lab2]
    
    print('Correlation Coefficient: ', corrCoeff)
    
    # Pull plot data
    data1 = df[lab1].dropna()
    t1 = data1.index
    #truncate the data based on bounds of data1
    df = df.truncate(before=data1.index[0], after = data1.index[-1])
    data2 = df[lab2]
    t2 = data2.index
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Years')
    ax1.set_ylabel(lab1, color=color)
    ax1.plot(t1, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) #cast x-axis ticks to ints
#    ax1.set_xlim(2013, 2016)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(lab2, color=color)  # we already handled the x-label with ax1
    ax2.plot(t2, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    print('PLOT 1') #DEBUG
    plt.show()
    
    if ' --' in lab2: #it's a time-lagged relationship
        # PLOT UNSHIFTED DATA 2 (to show 'forecast')
        data1 = df[lab1].dropna()
        #truncate the data based on data1
#        df = df.truncate(before=data1.index[0], after = data1.index[-1])
        data3 = full_df[lab2.split(' --')[0]] #get the unshifted data. will lag behind data1
        t3 = data3.index
        
        
        color = 'tab:red'
        ax1.plot(t1, data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
#        ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) #cast x-axis ticks to ints
        
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:blue'
        ax3.set_ylabel(lab2, color=color)  # we already handled the x-label with ax1
        ax3.plot(t3, data3, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        print('PLOT 2') #DEBUG
        plt.show() #<-- NOT WORKING RN

#comp_labels('Poland__Refugee Pop')
comp_labels('Russian Federation__Medical and health sciences R&D Fund')
full_df = full_fetch()    
    
#print(most_correlated('ANDORRA__Liquid Fuel Emissions', weighted = True))
    
#%%
#testbed
#0) find some data
x = full_df['Mauritius__Fertilizer Cons']

#1) Duplicate original df.
df2 = full_df.copy()

#2) Shift df2 index into past/future. Relabel columns to reflect time shift
#extract, turn to datetime, add year, turn back to index
years = df2.index
dt = pd.to_datetime(years, format = '%Y') #index of int -> datetime years
offset = .5 #years
t = 'year' if offset == 1 else 'years'
dt=pd.to_datetime(dt.year + offset, format='%Y')
df2.index = dt.year
tag = f' ({offset} {t} ahead)'
df2.columns =[col + tag for col in df2.columns.values] #rename for col

print(lagging_ind('Mauritius__Fertilizer Cons', 1), full_df)

#3) Re-merge, re-calculate
df3 = pd.merge(x, df2, how='inner', left_index = True, right_index=True)
