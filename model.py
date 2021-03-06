#!/usr/bin/env python
# coding: utf-8

# In[195]:


import pandas as pd
import numpy as np
import streamlit as st
#conda install statsmodels
#pip install statsmodels

from scipy import stats
import requests
import io
from sklearn.metrics import accuracy_score

import html5lib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from datetime import date
from datetime import datetime

import math


# In[412]:





##Loading in Regression and Classification Models
# model_reg = reg
# model_class = clf


# In[413]:


url_nfl = 'https://raw.githubusercontent.com/peanutshawny/nfl-sports-betting/master/data/spreadspoke_scores.csv'
url_teams = 'https://raw.githubusercontent.com/peanutshawny/nfl-sports-betting/master/data/nfl_teams.csv'
url_elo = 'https://raw.githubusercontent.com/peanutshawny/nfl-sports-betting/master/data/nfl_elo.csv'
stats = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/nfl_dataset_2002-2019week6.csv'

nfl_elo_latest = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo_latest.csv'

nfl_df = pd.read_csv(url_nfl, error_bad_lines=False)
teams_df = pd.read_csv(url_teams, error_bad_lines=False)
elo_df = pd.read_csv(url_elo, error_bad_lines=False)
stats_df = pd.read_csv(stats, error_bad_lines=False)
elo_latest = pd.read_csv(nfl_elo_latest, error_bad_lines=False)


# In[5]:


teams_df = teams_df[["team_name", "team_id"]]

teams_df['Nickname'] = teams_df["team_name"].str.split().str[-1]
teams_df = teams_df[(teams_df['team_name'] != 'Phoenix Cardinals') & (teams_df['team_name'] != 'St. Louis Cardinals') &
                   (teams_df['team_name'] != 'Boston Patriots') & (teams_df['team_name'] != 'Baltimore Colts') & (teams_df['team_name'] != 'Los Angeles Raiders') 
                   & (teams_df['Nickname'] != 'Oilers')]
#teams_df

#Merges
merged_df1 = nfl_df.merge(teams_df, left_on='team_home', right_on='team_name')
merged_df2 = merged_df1.merge(teams_df, left_on='team_away', right_on='team_name')
merged_df2 = merged_df2.rename(columns={"team_id_x": "home_id", "team_id_y": "away_id"})
merged_df2 = merged_df2.dropna(subset=['spread_favorite'])


# In[284]:


merged_df2['team_favorite_id'] = np.where(merged_df2['team_favorite_id']== 'PICK', merged_df2['home_id'] , merged_df2['team_favorite_id'])


# In[285]:


merged_df2['score_fav'] = np.where(merged_df2['team_favorite_id']== merged_df2['home_id'], merged_df2['score_home'] , merged_df2['score_away'])
merged_df2['score_underdog'] = np.where(merged_df2['team_favorite_id'] != merged_df2['home_id'], merged_df2['score_home'] , merged_df2['score_away'])
merged_df2['winner'] = np.where(merged_df2['score_fav'] > merged_df2['score_underdog'], 'Favorite', np.where(merged_df2['score_fav'] == merged_df2['score_underdog'], 'Tie', 'Underdog'))

merged_df2['indoor_outdoor'] = np.where(merged_df2['weather_detail'].isna(), 'Outdoors',
             np.where(merged_df2['weather_detail'].str.contains('DOME'), 'Indoors', 'Outdoors'))

merged_df2['schedule_week_numeric'] = np.where(merged_df2['schedule_week'] == 'Wildcard', 19, 
                                       np.where(merged_df2['schedule_week'] == 'Division', 20,
                                                np.where(merged_df2['schedule_week'] == 'Conference', 21,
                                                         np.where(merged_df2['schedule_week'] == 'Superbowl', 22,merged_df2['schedule_week']))))
#merged_df2               


# In[226]:


#merged_df2.isna().any()


# In[286]:


threshold_year = 2002
df3 = merged_df2[merged_df2['schedule_season'] > threshold_year]

elo_df3 = elo_df[(elo_df['season'] > threshold_year) & (elo_df['elo1_pre'].notna())]

#values:qbelo1_pre, qbelo2_pre

#elo_df3.columns.values.tolist()
#df3['schedule_date'] = df3['schedule_date'].values.astype('datetime64[M]')
#elo_df3['date'] = elo_df3['date'].values.astype('datetime64[M]')

#elo_df4 = pd.merge(df3, elo_df3,  how='inner', left_on=['schedule_date','home_id'], right_on = ['date','team1'])
#elo_df4


df3['schedule_date'] = pd.to_datetime(df3['schedule_date'])
elo_df3['date'] = pd.to_datetime(elo_df3['date'])
elo_df4 = pd.merge(df3, elo_df3,  how='inner', left_on=['schedule_date','home_id'], right_on = ['date','team1'])

#elo_df4


# In[287]:


elo_df4['elo_difference_homeaway'] = elo_df4['qbelo1_pre'] - elo_df4['qbelo2_pre']
elo_df4['elo_difference_fav_underdog']  = np.where(elo_df4['team_favorite_id'] == elo_df4['home_id'],
                                                   elo_df4['qbelo1_pre'], elo_df4['qbelo2_pre']) - np.where(elo_df4['team_favorite_id'] == elo_df4['home_id'],
                                                   elo_df4['qbelo2_pre'], elo_df4['qbelo1_pre'])

#elo_df4.sort_values(by=['schedule_date'])



# In[288]:


#Was the spread beaten? 
elo_df4['score_difference_fav_underdog'] = elo_df4['score_fav'] - elo_df4['score_underdog']
elo_df4['fav_beat_spread'] = np.where(elo_df4['score_difference_fav_underdog'] + elo_df4['spread_favorite'] > 0 , True, False)
elo_df4['fav_home'] = np.where(elo_df4['home_id'] == elo_df4['team_favorite_id'] , True, False)

elo_df5 = elo_df4.iloc[:, [0,1,2,3,4,5,6]]

#elo_df4.shape


# In[289]:


home_stats = stats_df.iloc[:, [0,2,10,9,12,11,14,13,16,15,18,17,20,19,26,25,28,36,37,38, 37]]
away_stats = stats_df.iloc[:, [0,1,9,10,11,12,13,14,15,16,17,18,19,20,25,26,27,35,36,37, 38]]

#home_stats

my_columns = ["date", "team", "passing_yards_team", "passing_yards_opp", "rushing_yards_team", "rushing_yards_opp",
             "total_yards_team", "total_yards_opp", "comp_att_team", "comp_att_opp", "sacks_team", "sacks_opp", "rushing_attempts_team",
             "rushing_attempts_opp", "turnovers_team", "turnovers_opp", "penalties_team", "possession_team", "possession_opp", "score_team", "score_opp"]


home_stats.columns = my_columns
away_stats.columns = my_columns

team_df1 = pd.concat([home_stats, away_stats])
team_df1 = team_df1.sort_values(["team", "date"], ascending = (True, True))





# In[290]:


seasons = elo_df4.iloc[:,[0,1]]
pd.set_option('display.max_rows', 500)
seasons.sort_values(by=['schedule_date'])

seasons = seasons.drop_duplicates()
seasons = seasons.sort_values("schedule_date")


team_df1['date'] = pd.to_datetime(team_df1['date'])
team_seasons = team_df1.merge(seasons, left_on='date', right_on='schedule_date')
team_seasons = team_seasons.sort_values(['team', 'date'], ascending = (True, True))
#team_seasons



#team_seasons['cumsum'] = team_seasons.groupby(['schedule_season'])['passing_yards_team'].cumsum()



#team_seasons.groupby(['team', 'schedule_season']).sum()
#team_seasons
pd.set_option('display.max_rows', 500)


ts_df = team_seasons

ts_df = ts_df.assign(passing_yards_team=ts_df.groupby(['team','schedule_season']).passing_yards_team.cumsum())
ts_df = ts_df.assign(passing_yards_opp=ts_df.groupby(['team','schedule_season']).passing_yards_opp.cumsum())
ts_df = ts_df.assign(rushing_yards_team=ts_df.groupby(['team','schedule_season']).rushing_yards_team.cumsum())
ts_df = ts_df.assign(rushing_yards_opp=ts_df.groupby(['team','schedule_season']).rushing_yards_opp.cumsum())
ts_df = ts_df.assign(total_yards_team=ts_df.groupby(['team','schedule_season']).total_yards_team.cumsum())
ts_df = ts_df.assign(total_yards_opp=ts_df.groupby(['team','schedule_season']).total_yards_opp.cumsum())


ts_df['team_completions'] = ts_df['comp_att_team'].apply(str).apply(lambda x: x.split('-')[0]).astype('int64')
ts_df['team_pass_attempts'] = ts_df['comp_att_team'].apply(str).apply(lambda x: x.split('-')[1]).astype('int64')

ts_df['opp_completions'] = ts_df['comp_att_opp'].apply(str).apply(lambda x: x.split('-')[0]).astype('int64')
ts_df['opp_pass_attempts'] = ts_df['comp_att_opp'].apply(str).apply(lambda x: x.split('-')[1]).astype('int64')

ts_df['team_sack_num'] = ts_df['sacks_team'].apply(str).apply(lambda x: x.split('-')[0]).astype('int64')
ts_df['team_sack_yds'] = ts_df['sacks_team'].apply(str).apply(lambda x: x.split('-')[1]).astype('int64')

ts_df['opp_sack_num'] = ts_df['sacks_opp'].apply(str).apply(lambda x: x.split('-')[0]).astype('int64')
ts_df['opp_sack_yds'] = ts_df['sacks_opp'].apply(str).apply(lambda x: x.split('-')[1]).astype('int64')




#team_seasons.assign(comp_att_te=team_seasons.groupby(['team','schedule_season']).total_yards_team.cumsum())
#team_seasons.assign(total_yards_opp=team_seasons.groupby(['team','schedule_season']).total_yards_opp.cumsum())



ts_df = ts_df.assign(rushing_attempts_team=ts_df.groupby(['team','schedule_season']).rushing_attempts_team.cumsum())
ts_df = ts_df.assign(rushing_attempts_opp=ts_df.groupby(['team','schedule_season']).rushing_attempts_opp.cumsum())
ts_df = ts_df.assign(turnovers_team=ts_df.groupby(['team','schedule_season']).turnovers_team.cumsum())
ts_df = ts_df.assign(turnovers_opp=ts_df.groupby(['team','schedule_season']).turnovers_opp.cumsum())
ts_df = ts_df.assign(score_team=ts_df.groupby(['team','schedule_season']).score_team.cumsum())
ts_df = ts_df.assign(score_opp=ts_df.groupby(['team','schedule_season']).score_opp.cumsum())


ts_df = ts_df.assign(team_completions=ts_df.groupby(['team','schedule_season']).team_completions.cumsum())
ts_df = ts_df.assign(team_pass_attempts=ts_df.groupby(['team','schedule_season']).team_pass_attempts.cumsum())
ts_df = ts_df.assign(opp_completions=ts_df.groupby(['team','schedule_season']).opp_completions.cumsum())
ts_df = ts_df.assign(opp_pass_attempts=ts_df.groupby(['team','schedule_season']).opp_pass_attempts.cumsum())
ts_df = ts_df.assign(team_sack_num=ts_df.groupby(['team','schedule_season']).team_sack_num.cumsum())
ts_df = ts_df.assign(team_sack_yds=ts_df.groupby(['team','schedule_season']).team_sack_yds.cumsum())
ts_df = ts_df.assign(opp_sack_num=ts_df.groupby(['team','schedule_season']).opp_sack_num.cumsum())
ts_df = ts_df.assign(opp_sack_yds=ts_df.groupby(['team','schedule_season']).opp_sack_yds.cumsum())

ts_df = ts_df.assign(games_played =ts_df.groupby(['team','schedule_season']).date.cumcount()+1)

#ts_df


# In[305]:


elo_data = elo_df4.iloc[:,[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,18,19,21,22,23,24,25,26,40,41,56,57,58,59,60,61,62]]

df_all_home = elo_data.merge(ts_df, left_on = ['schedule_date','Nickname_x'], right_on = ['date','team'])
df_all = df_all_home.merge(ts_df, left_on = ['schedule_date_x','Nickname_y'], right_on = ['date','team'])

df_agg = df_all


df_agg['passing_ypg_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['passing_yards_team_x']/df_agg['games_played_x'])-(df_agg['passing_yards_team_y']/df_agg['games_played_y']),
                                        (df_agg['passing_yards_team_y']/df_agg['games_played_y'])-(df_agg['passing_yards_team_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['passing_yards_team_x', 'passing_yards_team_y'])



df_agg['passing_ypg_opp_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['passing_yards_opp_x']/df_agg['games_played_x'])-(df_agg['passing_yards_opp_y']/df_agg['games_played_y']),
                                        (df_agg['passing_yards_opp_y']/df_agg['games_played_y'])-(df_agg['passing_yards_opp_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['passing_yards_opp_x', 'passing_yards_opp_y'])



df_agg['rushing_ypg_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['rushing_yards_team_x']/df_agg['games_played_x'])-(df_agg['rushing_yards_team_y']/df_agg['games_played_y']),
                                        (df_agg['rushing_yards_team_y']/df_agg['games_played_y'])-(df_agg['rushing_yards_team_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['rushing_yards_team_x', 'rushing_yards_team_y'])
                                      

df_agg['rushing_ypg_opp_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['rushing_yards_opp_x']/df_agg['games_played_x'])-(df_agg['rushing_yards_opp_y']/df_agg['games_played_y']),
                                        (df_agg['rushing_yards_opp_y']/df_agg['games_played_y'])-(df_agg['rushing_yards_opp_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['rushing_yards_opp_x', 'rushing_yards_opp_y'])


df_agg['passing_comp_pct_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['team_completions_x']/df_agg['team_pass_attempts_x'])-(df_agg['team_completions_y']/df_agg['team_pass_attempts_y']),
                                        (df_agg['team_completions_y']/df_agg['team_pass_attempts_y'])-(df_agg['team_completions_x']/df_agg['team_pass_attempts_x']))
df_agg = df_agg.drop(columns=['team_completions_x', 'team_completions_y', 'team_pass_attempts_x', 'team_pass_attempts_y', 
                    'comp_att_team_x', 'comp_att_team_y'])
                                      

df_agg['passing_comp_pct_diff_opp'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['opp_completions_x']/df_agg['opp_pass_attempts_x'])-(df_agg['opp_completions_y']/df_agg['opp_pass_attempts_y']),
                                        (df_agg['opp_completions_y']/df_agg['opp_pass_attempts_y'])-(df_agg['opp_completions_x']/df_agg['opp_pass_attempts_x']))
df_agg = df_agg.drop(columns=['opp_completions_x', 'opp_completions_y', 'opp_pass_attempts_x', 'opp_pass_attempts_y', 
                    'comp_att_opp_x', 'comp_att_opp_y'])


df_agg = df_agg.drop(columns = ['total_yards_team_x','total_yards_team_y','total_yards_opp_x','total_yards_opp_y'])




df_agg['sacks_per_game_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['team_sack_num_x']/df_agg['games_played_x'])-(df_agg['team_sack_num_y']/df_agg['games_played_y']),
                                        (df_agg['team_sack_num_y']/df_agg['games_played_y'])-(df_agg['team_sack_num_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['team_sack_num_x', 'team_sack_num_x'])



df_agg['sacks_per_game_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['opp_sack_num_x']/df_agg['games_played_x'])-(df_agg['opp_sack_num_y']/df_agg['games_played_y']),
                                        (df_agg['opp_sack_num_y']/df_agg['games_played_y'])-(df_agg['opp_sack_num_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['opp_sack_num_x', 'opp_sack_num_y'])

df_agg = df_agg.drop(columns=['sacks_team_x', 'sacks_team_y', 'sacks_opp_x', 'sacks_opp_y', 'penalties_team_x', 'penalties_team_y'])#, #'penalties_opp_x', 'penalties_opp_y'])

df_agg = df_agg.drop(columns=['rushing_attempts_team_x', 'rushing_attempts_opp_x','rushing_attempts_team_y','rushing_attempts_opp_y'])

#Divide By Zero Handling
df_agg['turnovers_team_x'] = np.where(df_agg['turnovers_team_x'] == 0, 1, df_agg['turnovers_team_x'])
df_agg['turnovers_team_y'] = np.where(df_agg['turnovers_team_y'] == 0, 1, df_agg['turnovers_team_y'])
                                        
                                        
df_agg['turnover_ratio_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['turnovers_opp_x']/df_agg['turnovers_team_x'])-(df_agg['turnovers_opp_y']/df_agg['turnovers_team_y']),
                                        (df_agg['turnovers_opp_y']/df_agg['turnovers_team_y'])-(df_agg['turnovers_opp_x']/df_agg['turnovers_team_x']))
df_agg = df_agg.drop(columns=['turnovers_opp_x', 'turnovers_opp_y','turnovers_team_x', 'turnovers_team_y'])


df_agg['ppg_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['score_team_x']/df_agg['games_played_x'])-(df_agg['score_team_y']/df_agg['games_played_y']),
                                        (df_agg['score_team_y']/df_agg['games_played_y'])-(df_agg['score_team_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['score_team_x', 'score_team_y'])

df_agg['opp_ppg_diff'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], 
                                        (df_agg['score_opp_x']/df_agg['games_played_x'])-(df_agg['score_opp_y']/df_agg['games_played_y']),
                                        (df_agg['score_opp_y']/df_agg['games_played_y'])-(df_agg['score_opp_x']/df_agg['games_played_x']))
df_agg = df_agg.drop(columns=['score_opp_x', 'score_opp_y'])


df_agg = df_agg.drop(columns=['weather_humidity'])                               

df_agg['fav_home'] = np.where(df_agg['team_favorite_id']==df_agg['home_id'], True, False)

df_agg.loc[(df_agg['schedule_week']).str.upper() == 'WILDCARD', 'schedule_week'] = 19
df_agg.loc[df_agg['schedule_week'].str.upper() == 'DIVISION', 'schedule_week'] = 20
df_agg.loc[df_agg['schedule_week'].str.upper() == 'CONFERENCE', 'schedule_week'] = 21
df_agg.loc[df_agg['schedule_week'].str.upper() == 'SUPERBOWL', 'schedule_week'] = 22

#df_agg.sort_values(by=['schedule_date_x'])


#pd.DataFrame(list(elo_df4.columns))


# In[312]:


df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
df_agg['weather_detail'] = np.where(df_agg['weather_detail'].isnull(), 'FAIR', df_agg['weather_detail'])
df_agg['weather_temperature'] = np.where(df_agg['weather_temperature'].isnull(), 72.0, df_agg['weather_temperature'])
df_agg['weather_wind_mph'] = np.where(df_agg['weather_wind_mph'].isnull(), 0.0, df_agg['weather_wind_mph'])
#df_agg = df_agg.dropna()
#df_agg
#df_agg.isna().any()


# In[311]:





# In[313]:



goal = np.array(df_agg['score_difference_fav_underdog'])
goal2 = np.array(np.where(df_agg['fav_beat_spread'] == True, 1, 0))

pred_df = df_agg.drop(columns = 'score_difference_fav_underdog')
pred_df = pred_df.drop(columns=['schedule_date_x', 'schedule_season_x', 'team_home',
                             'score_home', 'score_away', 'team_away', 'team_favorite_id',
                             'spread_favorite', 'home_id', 'Nickname_x', 'away_id', 'Nickname_y',
                             'score_fav', 'score_underdog', 'winner', 'score1','score2', 'qbelo1_pre', 'qbelo2_pre',
                             'fav_beat_spread', 'elo_difference_homeaway', 'fav_beat_spread','date_x',
                             'team_x', 'possession_team_x', 'possession_opp_x', 'schedule_date_y', 'schedule_season_y',
                             'team_sack_yds_y', 'date_y', 'games_played_x', 'games_played_y',
                             'team_y', 'possession_team_y', 'possession_opp_y', 'schedule_date_y', 'schedule_season_y', 'schedule_date',
                              'schedule_season', 'team_sack_num_y', 'opp_sack_yds_y', 'team_sack_yds_x', 'opp_sack_yds_x'])  


pred_df['schedule_week'] = pred_df['schedule_week'].astype(int)

pred_df = pd.get_dummies(pred_df)

column_list = list(df_agg.columns)



# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(pred_df, goal, test_size = 0.3, random_state = 15)


train_features_win, test_features_win, train_labels_win, test_labels_win = train_test_split(pred_df, goal2, test_size = 0.3, random_state = 15)




# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)



# In[ ]:





# In[323]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 15)
#rf2 = RandomForestRegressor(n_estimators = 100, random_state = 15)
# Train the model on training data

rf.fit(train_features, train_labels)
#rf2.fit(train_features_win, train_labels_win)



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification


# X_train, X_test, y_train, y_test = train_test_split(pred_df, goal, test_size=0.3) # 70% training and 30% test

# clf=RandomForestClassifier(n_estimators=100)

# #Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(X_train,y_train)

#y_pred=clf.predict(X_test)


#y_pred

# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[443]:


#Getting Feature Importance:

dd = pd.DataFrame(zip(train_features.columns,rf.feature_importances_))

dd.sort_values(1, ascending = False)

#plot(train_features, rf.feature_importances_)


# In[267]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')


# test_features['preds'] = np.array(predictions)

# test_features['errors'] = np.array(errors)


# In[329]:


#predictions


# In[325]:





# In[464]:




#-----CURRENT-----------------------

df_full_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/df_full.csv'
weather_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/weather_df.csv'

df_full =  pd.read_csv(df_full_url, index_col=0)
df_weather =  pd.read_csv(weather_url, index_col=0)


# In[465]:


current_week_num =0

season_start = datetime.strptime('2021-09-07', '%Y-%m-%d').date()

current_week_num = math.ceil(((date.today()-season_start).days/7)+.01)


# In[466]:


current_week_df = pd.read_csv('/Users/colebrandt/Documents/NFL_Predictor/Data/spread_df.csv')

visitor_list = current_week_df['Away']
home_list = current_week_df['Home']
spread_list = current_week_df['Spread']
fav_list = current_week_df['fav_team']


# In[467]:



##--------------------------------------------------------Application Displayed Portion-----------------------------------------

current_week_df = pd.read_csv('/Users/colebrandt/Documents/NFL_Predictor/Data/spread_df.csv')

visitor_list = current_week_df['Away']
home_list = current_week_df['Home']
gt_list = current_week_df['GameTime']
spread_list = current_week_df['Spread']
fav_list = current_week_df['fav_team']
und_list = current_week_df['und_team']
fav_team_spread_list = current_week_df['fav_spread']
result_scores = []

home_full_list = []



for g in range(0,len(visitor_list)):
    visitor = visitor_list[g]
    home = home_list[g]
    spread = spread_list[g]
    favorite = fav_list[g]
    und = und_list[g]
    gt = gt_list[g]
    home_team_full = ''
    away_team_full = ''
    fav_team_full = ''
    und_team_full = ''
    
    for tt in range(0, len(df_full['Team_x'])):
        if home.split()[-1] in df_full.iat[tt,0]:
            home_team_full = df_full.iat[tt,0]
        if visitor.split()[-1] in df_full.iat[tt,0]:
            away_team_full = df_full.iat[tt,0]
        if favorite.split()[-1] in df_full.iat[tt,0]:
            fav_team_full = df_full.iat[tt,0]
        if und.split()[-1] in df_full.iat[tt,0]:
            und_team_full = df_full.iat[tt,0]
            
            
    
    df_fav = df_full[df_full['Team_x'] == fav_team_full]
    df_und = df_full[df_full['Team_x'] == und_team_full]
    result_score = 0
    
    if(current_week_num) > 18:
        isplayoffs = True
    else:
        isplayoffs = False

    if (home_team_full in list(df_weather['Home_Team'])):
        df_weather = df_weather[df_weather['Home_Team'] == home_team_full]
        weather = df_weather.iat[0,2]
        temp = df_weather.iat[0,3]
        wind = df_weather.iat[0,4]    
    else:
        weather = 'FAIR'
        temp = 72.0
        wind = 0.0
    
    weather_detail_dome = 0
    weather_detail_fog = 0
    weather_detail_fair = 1
    weather_detail_rain = 0
    weather_detail_rf = 0
    weather_detail_snow = 0
    weather_detail_sf = 0
    weather_detail_sfr = 0
    ioi = 0
    ioo = 0
    

    if ('DOME' in weather):
        weather_detail_dome = 1
        ioi = 1
    if (('RAIN' in weather) or ('SHOWERS' in weather)):
        weather_detail_rain = 1
        ioo = 1
        weather_detail_fair = 0
    if (('SNOW' in weather) or ('FLURRIES' in weather)):
        weather_detail_snow = 1
        ioo = 1
        weather_detail_fair = 0
    if ('FOG' in weather):
        weather_detail_fog = 1
        ioo = 1
        weather_detail_fair = 0
    if (('FOG' in weather) and (('RAIN' in weather) or ('SHOWERS' in weather))):
        weather_detail_rf = 1
        ioo = 1
        weather_detail_fair = 0
    if (('FOG' in weather) and (('SNOW' in weather) or ('FLURRIES' in weather))):
        weather_detail_sf = 1
        ioo = 1
        weather_detail_fair = 0
    if ((('SNOW' in weather) or ('FLURRIES' in weather)) and (('HAIL' in weather) or ('FREEZING RAIN' in weather))):
        weather_detail_sfr = 1
        ioo = 1 
        weather_detail_fair = 0


    elo_diff = df_fav.iat[0,3] - df_und.iat[0,3]  

    if home_team_full == df_fav.iat[0,3]:
        fh = True
    else:
        fh = False
            
    pypg_diff = df_fav.iat[0,7] - df_und.iat[0,7]
    rypg_diff = df_fav.iat[0,8] - df_und.iat[0,8]
    PFpg_diff = df_fav.iat[0,4] - df_und.iat[0,4]
    PApg_diff = df_fav.iat[0,5] - df_und.iat[0,5]
    offsack_diff = df_fav.iat[0,9] - df_und.iat[0,9]
    TOMgn_diff = df_fav.iat[0,10] - df_und.iat[0,10]
    comppct_diff = df_fav.iat[0,6] - df_und.iat[0,6]
    comppct_def_diff = df_fav.iat[0,11] - df_und.iat[0,11]
    pypg_def_diff = df_fav.iat[0,12] - df_und.iat[0,12]
    rypg_def_diff = df_fav.iat[0,13] - df_und.iat[0,13]
    dsack_diff = df_fav.iat[0,14] - df_und.iat[0,14]

    pd.set_option('display.max_columns', 50)

    model_inputs = [current_week_num,
                   isplayoffs,
                    False,
                    temp,
                    wind,
                    elo_diff,
                    fh,
                    pypg_diff,
                    pypg_def_diff,
                    rypg_diff,
                    rypg_def_diff,
                    comppct_diff,
                    comppct_def_diff,
                    offsack_diff,
                    TOMgn_diff,
                    PFpg_diff,
                    PApg_diff,
                    weather_detail_dome,
                    0,
                    weather_detail_fair,
                    weather_detail_fog,
                    weather_detail_rain,
                    weather_detail_rf,
                    weather_detail_snow,
                    weather_detail_sf,
                    weather_detail_sfr,
                    ioi,
                    ioo
                   ]
    home_full_list.append(home_team_full)
    result_scores.append(round(rf.predict(pd.DataFrame(model_inputs).T)[0], 1))



# In[481]:


preds


# In[490]:


preds = pd.DataFrame(zip(visitor_list, home_list, home_full_list, fav_list, und_list, fav_team_spread_list, spread_list, result_scores, gt_list))
preds = preds.rename(columns = {0:'Visitor', 1: 'Home', 2: 'Home Full', 3: 'Favorite', 4: 'Underdog', 5: 'Team_Spread', 6: 'Spread', 7: 'Predicted_Difference', 8: 'Game Time'})

for sp in range(0,len(preds['Spread'])):
    if preds.iat[sp,6] == 'PK':
        preds.iat[sp,6] = 0.0
    else:
        preds['Spread'][sp] = float(preds['Spread'][sp])
        
preds['Spread'] = preds['Spread'].astype('float')

preds['Amount_Beat_Spread'] = round(preds['Predicted_Difference'] + preds['Spread'], 2)
preds['Spread_Beater'] = np.where(preds['Amount_Beat_Spread'] > 0, preds['Favorite'], np.where(preds['Amount_Beat_Spread'] < 0, preds['Underdog'], 'Push') )

preds.to_csv('/Users/colebrandt/Documents/NFL_Predictor/Data/preds.csv')


# In[491]:


preds


# In[492]:



if (len(visitor) > 2) & (len(home) > 2):
    if (home in list(df_weather['Home_Team'])):
        df_weather = df_weather[df_weather['Home_Team'] == home]
        st.header('Gametime Weather:')
        st.text('Weather: '  + str(df_weather.iat[0,2]))
        st.text('Temperature (degrees Fahrenheit): '  + str(df_weather.iat[0,3]))
        st.text('Wind (mph): ' + str(df_weather.iat[0,4]))
    else:
        st.text('Weather data not available for this game in week ' + str(current_week_num))
        


# In[493]:


df_display = df_full[df_full['Team_x'].isin([visitor,home])]
df_display = df_display.rename(columns={"Team_x": "Team Full Name", "G": "Games Played", 'Team_y': 'Nickname', 'adj_elo': 'QB-Adjusted ELO Rating'})

if ((len(home)> 1) & (len(visitor)> 1)):
    st.header('')
    st.header('Season Stats:')
    st.table(df_display)



# In[ ]:





# In[452]:


df_full


# In[ ]:




