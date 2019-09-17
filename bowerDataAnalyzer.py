import pandas as pd
import numpy as np
import os, re, pdb
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

#directories
cur_dir = os.getcwd()

data_dir = os.path.join(cur_dir, "masterSummarizedData_090619ZJ.xlsx")

#processing excel file
xls = pd.ExcelFile(data_dir)
dfTotal = pd.read_excel(xls, "Total", index_col = 0)
f1Total = dfTotal[dfTotal.iloc[:, 0].str.contains('F1')]
parTotal = dfTotal[~dfTotal.iloc[:, 0].str.contains('F1')]

dfDaily = pd.read_excel(xls, "Daily", index_col = 0)
f1Daily = dfDaily[dfDaily.iloc[:, 0].str.contains('F1')]
parDaily = dfDaily[~dfDaily.iloc[:, 0].str.contains('F1')]

dfHourly = pd.read_excel(xls, "Hourly", index_col = 0)
f1Hourly = dfHourly[dfHourly.iloc[:, 0].str.contains('F1')]
parHourly = dfHourly[~dfHourly.iloc[:, 0].str.contains('F1')]

dfDailyAvg = pd.read_excel(xls, "Daily Averages", index_col = 0)
f1DailyAvg = dfDailyAvg[dfDailyAvg.iloc[:, 0].str.contains('F1')]
parDailyAvg = dfDailyAvg[~dfDailyAvg.iloc[:, 0].str.contains('F1')]

dfHourlyAvg = pd.read_excel(xls, "Hourly Averages", index_col = 0)
f1HourlyAvg = dfHourlyAvg[dfHourlyAvg.iloc[:, 0].str.contains('F1')]
parHourlyAvg = dfHourlyAvg[~dfHourlyAvg.iloc[:, 0].str.contains('F1')]

f1_trial_names = [x for x in dfDaily.iloc[:, 0] if (x.find("F1") >= 0)]

par_trial_names = [x for x in dfDaily.iloc[:, 0] if (x.find("F1") < 0 and x.find("empty") < 0)]


#creating daily dataframes
dailyCols = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']

dailyVolumesF1 = pd.DataFrame(columns = dailyCols)

dailyBowersF1 = pd.DataFrame(columns = dailyCols)

dailyVolumesPar = pd.DataFrame(columns = dailyCols)

dailyBowersPar = pd.DataFrame(columns = dailyCols)


#creating hourly dataframes
hourlyCols = ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9', 'Hour 10', 'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20', 'Hour 21', 'Hour 22', 'Hour 23', 'Hour 24', 'Hour 25', 'Hour 26', 'Hour 27', 'Hour 28', 'Hour 29', 'Hour 30', 'Hour 31', 'Hour 32', 'Hour 33', 'Hour 34', 'Hour 35', 'Hour 36', 'Hour 37', 'Hour 38', 'Hour 39', 'Hour 40', 'Hour 41', 'Hour 42', 'Hour 43', 'Hour 44', 'Hour 45', 'Hour 46', 'Hour 47', 'Hour 48', 'Hour 49', 'Hour 50', 'Hour 51', 'Hour 52', 'Hour 53', 'Hour 54', 'Hour 55', 'Hour 56', 'Hour 57', 'Hour 58', 'Hour 59', 'Hour 60']

hourlyVolumesF1 = pd.DataFrame(columns = (['Trial'] + hourlyCols))

hourlyBowersF1 = pd.DataFrame(columns = (['Trial'] + hourlyCols))

hourlyVolumesPar = pd.DataFrame(columns = (['Trial'] + hourlyCols))

hourlyBowersPar = pd.DataFrame(columns = (['Trial'] + hourlyCols))


#aligning F1 trials by day/hour
for trial in set(f1_trial_names): #unique values

    #trims down f1Daily and f1Hourly to trialDaily and trialHourly
    trialDaily = f1Daily[f1Daily.iloc[:, 0].str.contains(trial)]
    trialDaily = trialDaily[['totalVolume', 'bowerIndex', 'bowerIndex_0.4', 'bowerIndex_0.8', 'bowerIndex_1.2']]
    trialHourly = f1Hourly[f1Hourly.iloc[:, 0].str.contains(trial)]
    trialHourly = trialHourly[['totalVolume', 'bowerIndex', 'bowerIndex_0.2', 'bowerIndex_0.4', 'bowerIndex_0.8', 'Day']]

    #finds building start day and removes prior ones
    day = 1
    for index, row in trialDaily.iterrows(): #row = dfDaily.iloc[index, :]
        if (row['bowerIndex'] == 0 or pd.isnull(row['bowerIndex'])):
            trialDaily = trialDaily.iloc[1:, :]
            day += 1
            continue
        else:
            break

    #trims down to 5 days if necessary
    if (len(trialDaily.index) > 5):
        trialDaily = trialDaily.iloc[:5, :]

    #finds building start hour and removes those prior
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

    #daily data
    trialDailyVolumeDict = {dailyCols[i]: (trialDaily.iloc[i]['totalVolume'] if i < trialDaily.shape[0] else np.NaN) for i in range(5)}
    trialDailyVolumeDict.update({'Trial': trial})

    trialDailyBowerDict = {dailyCols[i]: (trialDaily.iloc[i]['bowerIndex'] if i < trialDaily.shape[0] else np.NaN) for i in range(5)}
    trialDailyBowerDict.update({'Trial': trial})

    dailyVolumesF1 = dailyVolumesF1.append(trialDailyVolumeDict, ignore_index = True)
    dailyBowersF1 = dailyBowersF1.append(trialDailyBowerDict, ignore_index = True)

    #hourly data
    trialHourlyVolumeDict = {hourlyCols[i]: (trialHourly.iloc[i]['totalVolume'] if i < trialHourly.shape[0] else np.NaN) for i in range(60)}
    trialHourlyVolumeDict.update({'Trial': trial})

    trialHourlyBowerDict = {hourlyCols[i]: (trialHourly.iloc[i]['bowerIndex'] if i < trialHourly.shape[0] else np.NaN) for i in range(60)}
    trialHourlyBowerDict.update({'Trial': trial})

    hourlyVolumesF1 = hourlyVolumesF1.append(trialHourlyVolumeDict, ignore_index = True)
    hourlyBowersF1 = hourlyBowersF1.append(trialHourlyBowerDict, ignore_index = True)


#set index to trial name
dailyVolumesF1 = dailyVolumesF1.set_index('Trial')
dailyBowersF1 = dailyBowersF1.set_index('Trial')
hourlyVolumesF1 = hourlyVolumesF1.set_index('Trial')
hourlyBowersF1 = hourlyBowersF1.set_index('Trial')

#filter out zeros
dailyVolumesF1[dailyVolumesF1.iloc[:, :] == 0] = np.NaN
dailyBowersF1[dailyBowersF1.iloc[:, :] == 0] = np.NaN
hourlyVolumesF1[hourlyVolumesF1.iloc[:, :] == 0] = np.NaN
hourlyBowersF1[hourlyBowersF1.iloc[:, :] == 0] = np.NaN

#add mean rows
dailyVolumesF1.loc['Mean'] = dailyVolumesF1.mean()
dailyBowersF1.loc['Mean'] = dailyBowersF1.mean()
hourlyVolumesF1.loc['Mean'] = hourlyVolumesF1.mean()
hourlyBowersF1.loc['Mean'] = hourlyBowersF1.mean()

#add std rows
dailyVolumesF1.loc['STD'] = dailyVolumesF1.iloc[:-1].std(skipna = True)
dailyBowersF1.loc['STD'] = dailyBowersF1.iloc[:-1].std(skipna = True)
hourlyVolumesF1.loc['STD'] = hourlyVolumesF1.iloc[:-1].std(skipna = True)
hourlyBowersF1.loc['STD'] = hourlyBowersF1.iloc[:-1].std(skipna = True)


#aligning parent trials by day/hour
for trial in set(par_trial_names): #unique values

    #trims down parDaily and parHourly to trialDaily and trialHourly
    trialDaily = parDaily[parDaily.iloc[:, 0].str.contains(trial)]
    trialDaily = trialDaily[['totalVolume', 'bowerIndex', 'bowerIndex_0.4', 'bowerIndex_0.8', 'bowerIndex_1.2']]
    trialHourly = parHourly[parHourly.iloc[:, 0].str.contains(trial)]
    trialHourly = trialHourly[['totalVolume', 'bowerIndex', 'bowerIndex_0.2', 'bowerIndex_0.4', 'bowerIndex_0.8', 'Day']]

    #finds building start day and removes prior ones
    day = 1
    for index, row in trialDaily.iterrows(): #row = dfDaily.iloc[index, :]
        if (row['bowerIndex'] == 0 or pd.isnull(row['bowerIndex'])):
            trialDaily = trialDaily.iloc[1:, :]
            day += 1
            continue
        else:
            break

    #trims down to 5 days if necessary
    if (len(trialDaily.index) > 5):
        trialDaily = trialDaily.iloc[:5, :]

    #finds building start hour and removes those prior
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

    #daily data
    trialDailyVolumeDict = {dailyCols[i]: (trialDaily.iloc[i]['totalVolume'] if i < trialDaily.shape[0] else np.NaN) for i in range(5)}
    trialDailyVolumeDict.update({'Trial': trial})

    trialDailyBowerDict = {dailyCols[i]: (trialDaily.iloc[i]['bowerIndex'] if i < trialDaily.shape[0] else np.NaN) for i in range(5)}
    trialDailyBowerDict.update({'Trial': trial})

    dailyVolumesPar = dailyVolumesPar.append(trialDailyVolumeDict, ignore_index = True)
    dailyBowersPar = dailyBowersPar.append(trialDailyBowerDict, ignore_index = True)

    #hourly data
    trialHourlyVolumeDict = {hourlyCols[i]: (trialHourly.iloc[i]['totalVolume'] if i < trialHourly.shape[0] else np.NaN) for i in range(60)}
    trialHourlyVolumeDict.update({'Trial': trial})

    trialHourlyBowerDict = {hourlyCols[i]: (trialHourly.iloc[i]['bowerIndex'] if i < trialHourly.shape[0] else np.NaN) for i in range(60)}
    trialHourlyBowerDict.update({'Trial': trial})

    hourlyVolumesPar = hourlyVolumesPar.append(trialHourlyVolumeDict, ignore_index = True)
    hourlyBowersPar = hourlyBowersPar.append(trialHourlyBowerDict, ignore_index = True)


#set index to trial name
dailyVolumesPar = dailyVolumesPar.set_index('Trial')
dailyBowersPar = dailyBowersPar.set_index('Trial')
hourlyVolumesPar = hourlyVolumesPar.set_index('Trial')
hourlyBowersPar = hourlyBowersPar.set_index('Trial')

#filter out zeros
dailyVolumesPar[dailyVolumesPar.iloc[:, :] == 0] = np.NaN
dailyBowersPar[dailyBowersPar.iloc[:, :] == 0] = np.NaN
hourlyVolumesPar[hourlyVolumesPar.iloc[:, :] == 0] = np.NaN
hourlyBowersPar[hourlyBowersPar.iloc[:, :] == 0] = np.NaN

#add mean rows
dailyVolumesPar.loc['Mean'] = dailyVolumesPar.mean()
dailyBowersPar.loc['Mean'] = dailyBowersPar.mean()
hourlyVolumesPar.loc['Mean'] = hourlyVolumesPar.mean()
hourlyBowersPar.loc['Mean'] = hourlyBowersPar.mean()

#add std rows
dailyVolumesPar.loc['STD'] = dailyVolumesPar.iloc[:-1].std(skipna = True)
dailyBowersPar.loc['STD'] = dailyBowersPar.iloc[:-1].std(skipna = True)
hourlyVolumesPar.loc['STD'] = hourlyVolumesPar.iloc[:-1].std(skipna = True)
hourlyBowersPar.loc['STD'] = hourlyBowersPar.iloc[:-1].std(skipna = True)

#drop columns with only nan
dailyVolumesF1.dropna(axis = 1, how = 'all', inplace = True)
dailyBowersF1.dropna(axis = 1, how = 'all', inplace = True)
dailyVolumesPar.dropna(axis = 1, how = 'all', inplace = True)
dailyBowersPar.dropna(axis = 1, how = 'all', inplace = True)
hourlyVolumesF1 = pd.concat((hourlyVolumesF1.iloc[:, :50], hourlyVolumesF1.iloc[:, 50:].dropna(axis = 1, how = 'all', inplace = False)), axis = 1)
hourlyBowersF1 = pd.concat((hourlyBowersF1.iloc[:, :50], hourlyBowersF1.iloc[:, 50:].dropna(axis = 1, how = 'all', inplace = False)), axis = 1)
hourlyVolumesPar = pd.concat((hourlyVolumesPar.iloc[:, :50], hourlyVolumesPar.iloc[:, 50:].dropna(axis = 1, how = 'all', inplace = False)), axis = 1)
hourlyBowersPar = pd.concat((hourlyBowersPar.iloc[:, :50], hourlyBowersPar.iloc[:, 50:].dropna(axis = 1, how = 'all', inplace = False)), axis = 1)


#displaying data
'''
print('Daily Volumes F1: \n', dailyVolumesF1)

print('\n\n\n')
print('Daily Bowers F1: \n', dailyBowersF1)

print('\n\n\n')
print('Hourly Volumes F1 \n', hourlyVolumesF1)

print('\n\n\n')
print('Hourly Bowers F1: \n', hourlyBowersF1)

print('\n\n\n')
print('Daily Volumes Parentals: \n', dailyVolumesPar)

print('\n\n\n')
print('Daily Bowers Parentals: \n', dailyBowersPar)

print('\n\n\n')
print('Hourly Volumes Parentals: \n', hourlyVolumesPar)

print('\n\n\n')
print('Hourly Bowers Parentals: \n', hourlyBowersPar)
'''

#writing to excel file
with pd.ExcelWriter(os.path.join(cur_dir, 'stats.xlsx'), engine = "openpyxl") as writer:
    dailyVolumesF1.to_excel(writer, sheet_name = 'dailyVolumes')
    dailyBowersF1.to_excel(writer, sheet_name = 'dailyBowers')
    hourlyVolumesF1.to_excel(writer, sheet_name = 'hourlyVolumes')
    hourlyBowersF1.to_excel(writer, sheet_name = 'hourlyBowers')
    dailyVolumesPar.to_excel(writer, sheet_name = 'dailyVolumes', startrow = 13)
    dailyBowersPar.to_excel(writer, sheet_name = 'dailyBowers', startrow = 13)
    hourlyVolumesPar.to_excel(writer, sheet_name = 'hourlyVolumes', startrow = 13)
    hourlyBowersPar.to_excel(writer, sheet_name = 'hourlyBowers', startrow = 13)


#aovrm = AnovaRM(dailyVolumesF1, )

#To Do:

#calculate ANOVA, Tukey's HSD Test

#For each metric, we want the daily, as well as the averaged hourly for building hours on building days only.
