import pandas as pd
import numpy as np
import os, re, pdb
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels

cur_dir = os.getcwd()

data_dir = os.path.join(cur_dir, "masterSummarizedData_090619ZJ.xlsx")

xls = pd.ExcelFile(data_dir)
dfTotal = pd.read_excel(xls, "Total", index_col = 0)
f1Total = dfTotal[dfTotal.iloc[:, 0].str.contains('F1')]

dfDaily = pd.read_excel(xls, "Daily", index_col = 0)
f1Daily = dfDaily[dfDaily.iloc[:, 0].str.contains('F1')]

dfHourly = pd.read_excel(xls, "Hourly", index_col = 0)
f1Hourly = dfHourly[dfHourly.iloc[:, 0].str.contains('F1')]

dfDailyAvg = pd.read_excel(xls, "Daily Averages", index_col = 0)
f1DailyAvg = dfDailyAvg[dfDailyAvg.iloc[:, 0].str.contains('F1')]

dfHourlyAvg = pd.read_excel(xls, "Hourly Averages", index_col = 0)
f1HourlyAvg = dfHourlyAvg[dfHourlyAvg.iloc[:, 0].str.contains('F1')]

trial_names = [x for x in dfDaily.iloc[:, 0] if (x.find("F1") >= 0)]

#print(f1Daily.shape)
#print(f1Hourly.shape)

for trial in set(trial_names): #unique values

    trialDaily = f1Daily[f1Daily.iloc[:, 0].str.contains(trial)]
    trialDaily = trialDaily[['totalVolume', 'bowerIndex', 'bowerIndex_0.4', 'bowerIndex_0.8', 'bowerIndex_1.2']]
    trialHourly = f1Hourly[f1Hourly.iloc[:, 0].str.contains(trial)]
    trialHourly = trialHourly[['totalVolume', 'bowerIndex', 'bowerIndex_0.2', 'bowerIndex_0.4', 'bowerIndex_0.8', 'Day']]
    day = 1

    for index, row in trialDaily.iterrows(): #row = dfDaily.iloc[index, :]
        if (row['bowerIndex'] == 0 or pd.isnull(row['bowerIndex'])):
            trialDaily = trialDaily.iloc[1:, :]
            day += 1
            continue
        else:
            break

    #print('\nDay: ', day)

    #trims down to 5 days if necessary
    if (len(trialDaily.index) > 5):
        trialDaily = trialDaily.iloc[:5, :]

    for index, row in trialHourly.iterrows(): #row = dfDaily.iloc[index, :]
        if row['Day'] <  day:
            trialHourly = trialHourly.iloc[1:, :] #gets rid of one hour at a time when wrong day
            continue
        if row['Day'] == day:
            if (row['bowerIndex'] == 0 or pd.isnull(row['bowerIndex'])):
                trialHourly = trialHourly.iloc[1:, :] #gets rid of one hour at a time when building hasn't started yet
                continue
            else:
                break

    #trims down hours to 5 days after building
    trialHourly = trialHourly[trialHourly['Day'] <= (day + 4)]


#To Do:

#append dailys to larger graphing dataframe
#calculate mean/standard deviation per day

#append hourlys
#calculate ANOVA, Tukey's HSD Test

#For each metric, we want the daily, as well as the averaged hourly for building hours on building days only.
