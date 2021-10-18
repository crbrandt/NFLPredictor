#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import streamlit as st
#conda install statsmodels
#pip install statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import requests
import io
from sklearn.metrics import accuracy_score

import html5lib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


# In[42]:


##Updating Page Logo and Tab Title
st.set_page_config(page_title='NFL Game Predictor',
                   page_icon='https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png/revision/latest?cb=20120419223002',
                   layout="wide")

# div[data-baseweb="select"] > div {
#     background-color: '#575757';
# }

##Creating Text format options with orange colors
def highlight(text):
     st.markdown(f'<p style="text-align: center;color:#013369;font-size:22px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
def color(text):
     st.markdown(f'<p style="color:#013369;font-size:20px;border-radius:2%;">{text}</p>', unsafe_allow_html=True)


##Loading in Regression and Classification Models
# model_reg = reg
# model_class = clf


# In[4]:


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


# In[6]:


merged_df2['score_fav'] = np.where(merged_df2['team_favorite_id']== merged_df2['home_id'], merged_df2['score_home'] , np.where(merged_df2['spread_favorite'] != 0, merged_df2['score_away'], np.nan))
merged_df2['score_underdog'] = np.where(merged_df2['team_favorite_id']== merged_df2['home_id'], np.where(merged_df2['spread_favorite'] != 0, merged_df2['score_away'], np.nan), merged_df2['score_home'])
merged_df2['winner'] = np.where(merged_df2['score_fav'] > merged_df2['score_underdog'], 'Favorite', np.where(merged_df2['score_fav'] == merged_df2['score_underdog'], 'Tie', 'Underdog'))

merged_df2['indoor_outdoor'] = np.where(merged_df2['weather_detail'].isna(), 'Outdoors',
             np.where(merged_df2['weather_detail'].str.contains('DOME'), 'Indoors', 'Outdoors'))

merged_df2['schedule_week_numeric'] = np.where(merged_df2['schedule_week'] == 'Wildcard', 19, 
                                       np.where(merged_df2['schedule_week'] == 'Division', 20,
                                                np.where(merged_df2['schedule_week'] == 'Conference', 21,
                                                         np.where(merged_df2['schedule_week'] == 'Superbowl', 22,merged_df2['schedule_week']))))
#merged_df2               


# In[18]:


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


# In[19]:


elo_df4['elo_difference_homeaway'] = elo_df4['qbelo1_pre'] - elo_df4['qbelo2_pre']
elo_df4['elo_difference_fav_underdog']  = np.where(elo_df4['team_favorite_id'] == elo_df4['home_id'],
                                                   elo_df4['qbelo1_pre'], elo_df4['qbelo2_pre']) - np.where(elo_df4['team_favorite_id'] == elo_df4['home_id'],
                                                   elo_df4['qbelo2_pre'], elo_df4['qbelo1_pre'])

#elo_df4



# In[20]:


#Was the spread beaten? 
elo_df4['score_difference_fav_underdog'] = elo_df4['score_fav'] - elo_df4['score_underdog']
elo_df4['fav_beat_spread'] = np.where(elo_df4['score_difference_fav_underdog'] + elo_df4['spread_favorite'] > 0 , True, False)
elo_df4['fav_home'] = np.where(elo_df4['home_id'] == elo_df4['team_favorite_id'] , True, False)

elo_df5 = elo_df4.iloc[:, [0,1,2,3,4,5,6]]

#elo_df4.shape


# In[21]:


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





# In[22]:


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


# In[23]:


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

#df_agg


#pd.DataFrame(list(elo_df4.columns))


# In[33]:


df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
df_agg = df_agg.dropna()
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




print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[36]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 15)
rf2 = RandomForestClassifier(n_estimators = 1000, random_state = 15)
# Train the model on training data

rf.fit(train_features, train_labels)
rf2.fit(train_features_win, train_labels_win)



from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


X_train, X_test, y_train, y_test = train_test_split(pred_df, goal, test_size=0.3) # 70% training and 30% test

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


#y_pred

# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[26]:


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


# In[40]:


# Use the forest's predict method on the test data
predictions2 = rf2.predict(test_features_win)
# Calculate the absolute errors
#errors2 = abs(predictions2 - test_labels_win)

#accuracy_score(test_labels_win, predictions2)


# In[29]:





# In[30]:


df_full_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/df_full.csv'
weather_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/weather_df.csv'

df_full =  pd.read_csv(df_full_url, index_col=0)
df_weather =  pd.read_csv(weather_url, index_col=0)



# In[47]:


model_inputs = {
  'ppg_diff':0,
  'opp_ppg_diff':0,
  'turnover_ratio_diff':0,
  'elo_difference_fav_underdog':0,
  'passing_ypg_opp_diff':0,
  'passing_comp_pct_diff_opp':0,
  'rushing_ypg_opp_diff':0,
  'sacks_per_game_diff':0,
  'passing_ypg_diff':0,
  'rushing_ypg_diff':0,
  'passing_comp_pct_diff':0,
  'weather_temperature':0,
  'schedule_week':0,
  'weather_wind_mph':0,
  'fav_home':0,
  'weather_detail_DOME':0,
  'indoor_outdoor_Outdoors':0,
  'indoor_outdoor_Indoors':0,
  'weather_detail_Rain':0,
  'weather_detail_DOME (Open Roof)':0,
  'schedule_playoff':0,
  'weather_detail_Fog':0,
  'weather_detail_Rain | Fog':0,
  'weather_detail_Snow':0,
  'stadium_neutral':0,
  'weather_detail_Snow | Fog':0,
  'weather_detail_Snow | Freezing Rain':0
}

##Creating prediction functions
def predict(industry,moods,celebs):
    for industry_selection in industry:
        if industry_map[industry_selection] in model_inputs:
            model_inputs[industry_map[industry_selection]] = 1

    for mood in moods:
        if mood_map[mood] in model_inputs:
          model_inputs[mood_map[mood]] = 1
          
    # Interaction Inputs
    model_inputs['industry_mortgages * mood_funny'] = model_inputs['industry_mortgages'] * model_inputs['mood_funny']
    model_inputs['industry_cars1 * mood_funny'] = model_inputs['industry_cars1'] * model_inputs['mood_funny']

    prediction = round(model.predict(model_inputs)[0]*100,2)
    if prediction < 0:
      prediction = 0
    if prediction > 100:
      prediction = 100
    
    return prediction

###Mapping industry values in the data to their display options on the application
industry_map = {
    'Auto Parts & Accessories':'industry_auto_parts_accessories',
    'Airlines':'industry_airline_industry',
    'Beauty':'industry_beauty',
    'Beer':'industry_beer',
    'Beverages':'industry_beverages',
    'Candy':'industry_candy',
    'Cars':'industry_cars1',
    'Cellular, Internet, and TV Providers': 'industry_cellular',
    'Cleaning Supplies':'industry_cleaning_supplies',
    'Cola Drinks': 'industry_cola_drinks',
    'Software and Technology':'industry_computer_software',
    'Computer Hardware':'industry_computer_hardware',
    'Credit Cards':'industry_credit_cards',
    'Deodorant':'industry_deodorant',
    'Dips':'industry_dips',
    'Music, Movies, and Entertainment':'industry_entertainment',
    'Energy Drinks':'industry_energy_drinks',
    'Restaurants and Fast Food':'industry_fast_food',
    'Financial Services_1':'industry_financial_services',
    'Food Delivery':'industry_food_delivery',
    'Freelancers':'industry_freelancers',
    'Games':'industry_games',
    'Home Security':'industry_home_security',
    'Hotels':'industry_hotels',
    'Hygiene':'industry_hygiene',
    'Insurance':'industry_insurance',
    'Investments':'industry_investments',
    'Job Search':'industry_job_search',
    'Lawn Care':'industry_lawn_care',
    'Liquors':'industry_alcoholic_beverages',
    'Financial Services':'industry_loans',
    'Mobile Phones':'industry_mobile_phones',
    'Mortgages':'industry_mortgages',
    'Movies':'industry_movies',
    'Nuts':'industry_nuts',
    'Online Retailers':'industry_online_retailers',
    'Online Streaming Services':'industry_online_streaming',
    'Pizza':'industry_pizza',
    'Potato Chips':'industry_potato_chips',
    'Retail Stores':'industry_retail_stores',
    'Search Engines':'industry_search_engines',
    'Shoes':'industry_shoes',
    'Snacks':'industry_snacks1',
    'Soap':'industry_soap',
    'Social Media':'industry_social_media',
    'Soft Drinks':'industry_soft_drinks',
    'Sports Leagues':'industry_sports_leagues',
    'Taxes':'industry_taxes',
    'Travel':'industry_travel_industry',
    'Trucks':'industry_trucks',
    'TV Providers':'industry_TV_providers',
    'Virtual Assistants':'industry_virtual_assistants',
    'Water':'industry_water',
    'Yogurt':'industry_yogurt',
    'Other':'other'
    }

mood_map = {
    'Adventurous':'mood_adventurous',
    'Alluring':'mood_alluring',
    'Boring':'mood_boring',
    'Controversial':'mood_controversial',
    'Cute/Adorable':'mood_cute\adorable',
    'Dramatic':'mood_dramatic',
    'Emotional':'mood_emotional',
    'Exciting':'mood_exciting',
    'Flirty':'mood_flirty',
    'Funny':'mood_funny',
    'Goofy':'mood_goofy',
    'Gross':'mood_gross',
    'Heartwarming':'mood_heartwarming',
    'Informative':'mood_informative',
    'Inspirational':'mood_inspirational',
    'Light-hearted':'mood_light hearted',
    'Mysterious':'mood_mysterious',
    'Party-themed':'mood_party themed',
    'Patriotic':'mood_patriotic',
    'Romantic':'mood_romantic',
    'Scary':'mood_scary',
    'Serious':'mood_serious',
    'Sexy':'mood_sexy',
    'Shocking':'mood_shocking',
    'Somber':'mood_somber',
    'Suspenseful':'mood_suspenseful',
    'Unique':'mood_unique',
    'Weird':'mood_weird'
    }

celeb_map = {
    'Athletes':'n_athlete',
    'Bands':'n_band',
    'Business Leaders':'n_business_leader',
    'Comedians':'n_comedian',
    'Football Coaches':'n_football_coaches',
    'Historical Figures':'n_historical_figures',
    'Models':'n_models',
    'Musicians':'n_musician',
    'NFL Players':'n_nfl',
    'Politicians':'n_politician',
    'Reality TV Stars':'n_reality_tv_stars',
    'Sports Commentators':'n_sports_commentators',
    'Talk Show Hosts':'n_talk_show_hosts',
    'Top Actors':'n_top_actors'
    }



##--------------------------------------------------------Application Displayed Portion-----------------------------------------

##Header and Logo
col_title, col_logo = st.beta_columns([4,1])
with col_title:
  st.title('NFL Game Predictor')
  st.markdown(' ## Created by Cole Brandt')
with col_logo:
  st.image("https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png")
st.write("#")


col1, col2, col3 = st.beta_columns(3)
visitor = ''
home = ''

with col1:
    st.header("Visiting Team")
    visitor = st.selectbox('Select the visiting team', df_full['Team_x'].sort_values())
    st.image("https://static.streamlit.io/examples/cat.jpg")
with col2:
    st.markdown("<h1 style='text-align: center; color: red;'>vs.</h1>", unsafe_allow_html=True)
with col3:
    st.header("Home Team")
    home = st.selectbox('Select the home team', df_full[df_full['Team_x'] != visitor]['Team_x'].sort_values())
    st.image("https://static.streamlit.io/examples/cat.jpg")

##Selectbox for Favorite
away = st.multiselect(
      'Select the Away Team',
      (
          list(df_full['Team_x'].sort())
      ), help = 'Please select the road team for the game.'
      )

if len(favorite) > 1:
  color('Please limit your selection to one team.')

##Selectbox for underdog (note: if more than one mood is selected, users will not be able to continue)
underdog = st.multiselect(
    'Select the Underdog',
    list(df_full['Team_x'].sort()), help = 'Please select the team which is not favored in the game.'
    )

if len(underdog) > 1:
  color('Please limit your mood selections to one team.')


# # ##Multiselect for celebrity types. When celebrities are selected, their specific count values will be editable using sliders.
# # celebs = st.multiselect(
# #     'Select Types of Celebrities',
# #     [x for x in celeb_map.keys()], help = 'Please select the type(s) of celebrities which will be included in your advertisement.'
# #     )

# # celeb_sliders = []
# # for celeb in celebs:
# #     
# spread_slider = st.slider('Spread' , min_value=-30.0, max_value=30.0, step = 0.5)

# spread = spread_slider
    

# ##Predictions for score and cluster
# if len(favorite) <= 1 and len(underdog) <= 1:
#     button = st.button('Predict')
  

# # Bottom info bar ------------------------------------------------------------------------
# st.markdown('___')
# about = st.beta_expander('About')
# with about:
#     '''
#     Thank you for visiting the Super Bowl Advertisement Optimizer, powered by Caryt Marketing Co. For more information, please visit our team's [Github repository] (https://github.com/crbrandt/CarytMarketingCo).
    
#     Curated by the Caryt Marketing Co. Analytics team: \n
#     Cole Brandt, Anton Averin, Ranaa Ansari, Young Jang, Tyrone Brown
    
#     [Contact Our Team] (mailto:colebrandt2021@u.northwestern.edu)
#     '''
    
# st.image("https://i.ibb.co/9qDzx87/Sunrise-Abstract-Shapes-Logo-Template-copy.png",
#     width= 100, caption='2021 Caryt Marketing Co.')


# In[ ]:




