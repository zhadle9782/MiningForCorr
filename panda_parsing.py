# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:25:19 2020

@author: Zion
"""
#%%
#Initialization
#import sys


from datapackage import Package
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

package = Package('https://datahub.io/core/co2-fossil-by-nation/datapackage.json')

resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
#        print(data.head(400))

#%%
'''
GOAL: Partition dataframe into a df for each data-type
    * df struct: index = year, col = country
    * df content: data-type
'''


#1) get full list of country names
data_url = 'https://datahub.io/core/country-list/datapackage.json'

# to load Data Package into storage
package = Package(data_url)

# to load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data2 = pd.read_csv(resource.descriptor['path'])
#        print (datapr)
countries = data2.iloc[:,0].to_list()

#2) iterate over and partition df into dfs by country
cframes = []
for country in countries:
    df = data[data['Country'] == country.upper()]
    if not df.empty:
        cframes.append(df)

#3) extract each type of data from each df and build desired output dfs

#a) for each type of data in the column, merge into a new df, 
#where column name is country

data_types = data.columns.to_list() #get data labels
partitions = {}  #list of dfs organized by data type; type:df
#iterate over each data type
for i in range(2, len(data_types)): #starts at  bc we don't need country name & year
    dtype = data_types[i]
    merge_df = None #build up later
    for country in cframes:
        cdata = country.iloc[:, [0,i]]
        cname = country.iloc[0, 1]
        cdata = cdata.rename(columns = {dtype: cname})
        if merge_df is None: #start a new merge_df
            merge_df = cdata
        else:
            merge_df = pd.merge(merge_df, cdata, how='outer', on='Year')
    partitions[dtype] = merge_df.sort_values('Year')

g = partitions['Total'].plot(x='Year', title='Total C02 Emissions', figsize = (8,8))
g.set_xlabel('Years (till 2014)')
    

#%%
#Visualizing
            
for field, frame in partitions.items():
    try:
        graph = frame.plot(x='Year', title= field, figsize = (16,8))
        graph.set_xlabel('Year')
        print('Plotting... ', field)
    except:
        print('This part is buggy: ', field)

#print(partitions['Total'])
