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


# In[335]:


##Updating Page Logo and Tab Title
st.set_page_config(page_title='NFL Game Predictor',
                   page_icon='https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png',
                   layout="wide")

# div[data-baseweb="select"] > div {
#     background-color: '#575757';
# }

##Creating Text format options with base and team colors
def highlight(text):
     st.markdown(f'<p style="text-align: center;color:#013369;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
def teamcolor(text):
     if 'Arizona' in text:
         st.markdown(f'<p style="text-align: center;color:#97233F;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Atlanta' in text:
         st.markdown(f'<p style="text-align: center;color:#A71930;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Baltimore' in text:
         st.markdown(f'<p style="text-align: center;color:#241773;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Buffalo' in text:
         st.markdown(f'<p style="text-align: center;color:#00338D;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Carolina' in text:
         st.markdown(f'<p style="text-align: center;color:#0085CA;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Chicago' in text:
         st.markdown(f'<p style="text-align: center;color:#C83803;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Cincinnati' in text:
         st.markdown(f'<p style="text-align: center;color:#FB4F14;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Cleveland' in text:
         st.markdown(f'<p style="text-align: center;color:#FF3C00;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Dallas' in text:
         st.markdown(f'<p style="text-align: center;color:#003594;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Denver' in text:
         st.markdown(f'<p style="text-align: center;color:#FB4F14;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Detroit' in text:
         st.markdown(f'<p style="text-align: center;color:#0076B6;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Green Bay' in text:
         st.markdown(f'<p style="text-align: center;color:#203731;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Houston' in text:
         st.markdown(f'<p style="text-align: center;color:#A71930;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Indianapolis' in text:
         st.markdown(f'<p style="text-align: center;color:#002C5F;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Jacksonville' in text:
         st.markdown(f'<p style="text-align: center;color:#006778;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Kansas City' in text:
         st.markdown(f'<p style="text-align: center;color:#E31837;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Las Vegas' in text:
         st.markdown(f'<p style="text-align: center;color:#A5ACAF;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Chargers' in text:
         st.markdown(f'<p style="text-align: center;color:#0080C6;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Rams' in text:
         st.markdown(f'<p style="text-align: center;color:#003594;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Miami' in text:
         st.markdown(f'<p style="text-align: center;color:#008E97;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Minnesota' in text:
         st.markdown(f'<p style="text-align: center;color:#4F2683;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'New England' in text:
         st.markdown(f'<p style="text-align: center;color:#002244;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'New Orleans' in text:
         st.markdown(f'<p style="text-align: center;color:#D3BC8D;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Giants' in text:
         st.markdown(f'<p style="text-align: center;color:#0B2265;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Jets' in text:
         st.markdown(f'<p style="text-align: center;color:#125740;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Eagles' in text:
         st.markdown(f'<p style="text-align: center;color:#004C54;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Steelers' in text:
         st.markdown(f'<p style="text-align: center;color:#FFB612;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'San Francisco' in text:
         st.markdown(f'<p style="text-align: center;color:#AA0000;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Seahawks' in text:
         st.markdown(f'<p style="text-align: center;color:#69BE28;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Buccaneers' in text:
         st.markdown(f'<p style="text-align: center;color:#D50A0A;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Titans' in text:
         st.markdown(f'<p style="text-align: center;color:#4B92DB;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     elif 'Washington' in text:
         st.markdown(f'<p style="text-align: center;color:#773141;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
     else:
         st.markdown(f'<p style="text-align: center;color:#575757;font-size:26px;border-radixus:2%;">{text}</p>', unsafe_allow_html=True)
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


# In[328]:


#predictions2 = rf2.predict(test_features_win)
#predictions2


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





# In[270]:


df_full_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/df_full.csv'
weather_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/weather_df.csv'

df_full =  pd.read_csv(df_full_url, index_col=0)
df_weather =  pd.read_csv(weather_url, index_col=0)


# In[271]:


current_week_num =0

season_start = datetime.strptime('2021-09-07', '%Y-%m-%d').date()

current_week_num = math.ceil(((date.today()-season_start).days/7)+.01)


# In[ ]:





# In[333]:



##--------------------------------------------------------Application Displayed Portion-----------------------------------------

##Header and Logo
col_title, col_logo = st.beta_columns([4,1])
with col_title:
  st.title('NFL Game Predictor')
  st.markdown(' ## Created by Cole Brandt')
  st.markdown('  Last updated: Tuesday, October 19th, 2021')  
with col_logo:
  st.image("https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png")
st.write("#")
highlight('NFL Week ' + str(current_week_num))



# home = 'Pittsburgh Steelers'
# visitor = 'Seattle Seahawks'




with st.form(key='fav_form'):
        
    visitor = ['Cole']
    home = ['Brandt']

    pic_home = 'https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png'
    pic_vis = 'https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png'

    df_full = df_full.sort_values(by=['Team_x'])
    df_full = df_full.reset_index(drop=True)


    # home = 'Pittsburgh Steelers'

    # visitor = 'Seattle Seahawks'
    col1, col2, col3 = st.beta_columns([3,1,3])

    with col1:
        st.markdown("<h1 style='text-align: center;'>Visiting Team</h1>", unsafe_allow_html=True)
        visitor = st.selectbox('Select the visiting team', ([' '] + list(df_full['Team_x'])))
        if 'Arizona Cardinals' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/177/full/kwth8f1cfa2sch5xhjjfaof90.png'
        elif 'Atlanta Falcons' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/173/full/299.png'
        elif 'Baltimore Ravens' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/153/full/318.png'
        elif 'Buffalo Bills' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/149/full/n0fd1z6xmhigb0eej3323ebwq.png'   
        elif 'Carolina Panthers' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/174/full/f1wggq2k8ql88fe33jzhw641u.png'    
        elif 'Chicago Bears' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/169/full/364.png'    
        elif 'Cincinnati Bengals' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/154/full/cincinnati_bengals_logo_primary_2021_sportslogosnet-2049.png'
        elif 'Cleveland Browns' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/155/full/7855_cleveland_browns-primary-2015.png'
        elif 'Dallas Cowboys' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/165/full/406.png'
        elif 'Denver Broncos' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/161/full/9ebzja2zfeigaziee8y605aqp.png'
        elif 'Detroit Lions' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/170/full/1398_detroit_lions-primary-2017.png'
        elif 'Green Bay Packers' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/171/full/dcy03myfhffbki5d7il3.png'
        elif 'Houston Texans' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/157/full/570.png'
        elif 'Indianapolis Colts' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/158/full/593.png'
        elif 'Jacksonville Jaguars' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/159/full/8856_jacksonville_jaguars-alternate-2013.png'   
        elif 'Kansas City Chiefs' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/162/full/857.png'    
        elif 'Las Vegas Raiders' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/6708/full/8521_las_vegas_raiders-primary-20201.png'    
        elif 'Los Angeles Chargers' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/6446/full/1660_los_angeles__chargers-primary-20201.png'
        elif 'Los Angeles Rams' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/5941/full/8334_los_angeles_rams-primary-20201.png'
        elif 'Miami Dolphins' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/150/full/7306_miami_dolphins-primary-2018.png'
        elif 'Minnesota Vikings' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/172/full/2704_minnesota_vikings-primary-20131.png'
        elif 'New England Patriots' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/151/full/y71myf8mlwlk8lbgagh3fd5e0.png'
        elif 'New Orleans Saints' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/175/full/907.png'
        elif 'New York Giants' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/166/full/80fw425vg3404shgkeonlmsgf.png'
        elif 'New York Jets' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/152/full/9116_new_york_jets-primary-2019.png'
        elif 'Philadelphia Eagles' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/167/full/960.png'   
        elif 'Pittsburgh Steelers' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/156/full/970.png'    
        elif 'San Francisco 49ers' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/179/full/9455_san_francisco_49ers-primary-2009.png'    
        elif 'Seattle Seahawks' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/180/full/pfiobtreaq7j0pzvadktsc6jv.png'
        elif 'Tampa Bay Buccaneers' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/176/full/8363_tampa_bay_buccaneers-primary-2020.png'
        elif 'Tennessee Titans' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/160/full/1053.png'
        elif 'Washington Football Team' in visitor:
            pic_vis = 'https://content.sportslogos.net/logos/7/6741/full/8837_washington_football_team-wordmark-20201.png'
        if len(visitor)> 1:
            st.image(pic_vis, height = 200)
        favorite = ''
        st.header(' ')
        if len(home) < 3:
            st.header(' ')
            st.header(' ')
        st.markdown('Which team is favored to win?')
        favorite = st.selectbox('Select the favorite', [' ','Visitor', 'Home'])
        st.markdown('')
        st.markdown('What is the spread?')
        spread = abs(st.number_input('Insert a number', min_value = -30.0, max_value = 30.0, value = 0.0, step = 0.5))
        st.header(' ')
        st.header(' ')

    with col2:
        st.markdown("<h1 style='text-align: center;'>vs.</h1>", unsafe_allow_html=True)

    with col3:
        st.markdown("<h1 style='text-align: center;'>Home Team</h1>", unsafe_allow_html=True)
        home = st.selectbox('Select the home team', ([' '] + list(df_full['Team_x'])))
        if 'Arizona Cardinals' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/177/full/kwth8f1cfa2sch5xhjjfaof90.png'
        elif 'Atlanta Falcons' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/173/full/299.png'
        elif 'Baltimore Ravens' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/153/full/318.png'
        elif 'Buffalo Bills' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/149/full/n0fd1z6xmhigb0eej3323ebwq.png'   
        elif 'Carolina Panthers' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/174/full/f1wggq2k8ql88fe33jzhw641u.png'    
        elif 'Chicago Bear' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/169/full/364.png'    
        elif 'Cincinnati Bengals' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/154/full/cincinnati_bengals_logo_primary_2021_sportslogosnet-2049.png'
        elif 'Cleveland Browns' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/155/full/7855_cleveland_browns-primary-2015.png'
        elif 'Dallas Cowboys' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/165/full/406.png'
        elif 'Denver Broncos' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/161/full/9ebzja2zfeigaziee8y605aqp.png'
        elif 'Detroit Lions' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/170/full/1398_detroit_lions-primary-2017.png'
        elif 'Green Bay Packers' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/171/full/dcy03myfhffbki5d7il3.png'
        elif 'Houston Texans' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/157/full/570.png'
        elif 'Indianapolis Colts' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/158/full/593.png'
        elif 'Jacksonville Jaguars' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/159/full/8856_jacksonville_jaguars-alternate-2013.png'   
        elif 'Kansas City Chiefs' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/162/full/857.png'    
        elif 'Las Vegas Raiders' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/6708/full/8521_las_vegas_raiders-primary-20201.png'    
        elif 'San Diego Chargers' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/6446/full/1660_los_angeles__chargers-primary-20201.png'
        elif 'Los Angeles Rams' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/5941/full/8334_los_angeles_rams-primary-20201.png'
        elif 'Miami Dolphins' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/150/full/7306_miami_dolphins-primary-2018.png'
        elif 'Minnesota Vikings' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/172/full/2704_minnesota_vikings-primary-20131.png'
        elif 'New England Patriots' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/151/full/y71myf8mlwlk8lbgagh3fd5e0.png'
        elif 'New Orleans Saints' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/175/full/907.png'
        elif 'New York Giants' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/166/full/80fw425vg3404shgkeonlmsgf.png'
        elif 'New York Jets' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/152/full/9116_new_york_jets-primary-2019.png'
        elif 'Philadelphia Eagles' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/167/full/960.png'   
        elif 'Pittsburgh Steelers' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/156/full/970.png'    
        elif 'San Francisco 49ers' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/179/full/9455_san_francisco_49ers-primary-2009.png'    
        elif 'Seattle Seahawks' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/180/full/pfiobtreaq7j0pzvadktsc6jv.png'
        elif 'Tampa Bay Buccaneers' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/176/full/8363_tampa_bay_buccaneers-primary-2020.png'
        elif 'Tennessee Titans' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/160/full/1053.png'
        elif 'Washington Football Team' in home:
            pic_home = 'https://content.sportslogos.net/logos/7/6741/full/8837_washington_football_team-wordmark-20201.png'
        if len(home)> 1:
            st.image(pic_home, height = 200)            

    
    c1, c2, c3 = st.beta_columns([3.3,1,3])
    with c1:
        st.text('')
    with c2:
        submit_button = st.form_submit_button(label='Predict Result')
    with c3:
        st.text('')

        
# favorite = 'Pittsburgh Steelers'
# spread = 5.0
 
if favorite == 'Visitor':
    favorite = visitor
elif favorite == 'Home':
    favorite = home
else:
    favorite = ''



# In[316]:


df_fav = df_full[df_full['Team_x'] == favorite]
df_und = df_full[(df_full['Team_x'].isin([visitor,home])) & (df_full['Team_x'] != favorite)]
result_score = 0
result_prob = 0

if(current_week_num) > 18:
    isplayoffs = True
else:
    isplayoffs = False
    
    
if (home in list(df_weather['Home_Team'])):
        df_weather = df_weather[df_weather['Home_Team'] == home]
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
    
    
if (len(favorite) > 2):
    elo_diff = df_fav.iat[0,3] - df_und.iat[0,3]  

    if home == df_fav.iat[0,3]:
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
    
    result_score = rf.predict(pd.DataFrame(model_inputs).T)
    #result_beat_spread = rf2.predict(pd.DataFrame(model_inputs).T)
    #pred_df
    #df_fav
    #model_inputs
    #df_fav


# In[274]:


#df_full[(df_full['Team_x'].isin([visitor,home])) & (df_full['Team_x'] != favorite)]
#df_full[(df_full['Team_x'].isin([visitor,home])) & (df_full['Team_x'] != favorite)].iat[0,0]
# st.markdown('Favorite: ' + favorite)
# st.markdown('Home: ' + home)
# st.markdown('Visitor: ' + visitor)


# In[336]:


if len(favorite) > 2:
    underdog = df_full[(df_full['Team_x'].isin([visitor,home])) & (df_full['Team_x'] != favorite)].iat[0,0]
    if round(result_score[0], 2) > 0.0:
        teamcolor('The ' + favorite + ' are projected to win by ' + str(round(result_score[0], 2)) + ' points') 
    elif round(result_score[0], 2) < 0.0:
        teamcolor('The ' + underdog + ' are projected to win by ' + str(-1 * round(result_score[0],2)) + ' points') 
    else:
        highlight('The ' + favorite + ' and the ' + underdog + ' are expected to tie') 

    beat_spread_amt = result_score[0] - abs(spread)
    if beat_spread_amt > 0 :
        teamcolor('The ' + favorite + ' would beat the spread by ' + str(round(beat_spread_amt, 2)) + ' points') 
    elif beat_spread_amt < 0.0:
        teamcolor('The ' + underdog + ' would beat the spread by ' + str(abs(round(beat_spread_amt, 2))) + ' points')   
    else:
        highlight('The ' + favorite + ' and the ' + underdog + ' would push') 


# In[318]:



if (len(visitor) > 2) & (len(home) > 2):
    if (home in list(df_weather['Home_Team'])):
        df_weather = df_weather[df_weather['Home_Team'] == home]
        st.header('Gametime Weather:')
        st.text('Weather: '  + str(df_weather.iat[0,2]))
        st.text('Temperature (degrees Fahrenheit): '  + str(df_weather.iat[0,3]))
        st.text('Wind (mph): ' + str(df_weather.iat[0,4]))
    else:
        st.text('Weather data not available for this game in week ' + str(current_week_num))
        


# In[319]:


df_display = df_full[df_full['Team_x'].isin([visitor,home])]
df_display = df_display.rename(columns={"Team_x": "Team Full Name", "G": "Games Played", 'Team_y': 'Nickname', 'adj_elo': 'QB-Adjusted ELO Rating'})

if ((len(home)> 1) & (len(visitor)> 1)):
    st.header('')
    st.header('Season Stats:')
    st.table(df_display)



# In[320]:


# Bottom info bar ------------------------------------------------------------------------
st.markdown('___')
about = st.beta_expander('About')
with about:
    '''
    Thank you for visiting the NFL Game Predictor, developed by Cole Brandt. For more information, please visit my [Github repository] (https://github.com/crbrandt/NFLPredictor).
    
    All images sourced from sportslogos.net
    
    [Contact Me] (mailto:cole.r.brandt@gmail.com)
    '''
    
st.image("https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png",
    width= 100, caption='2021 Cole Brandt')


# In[ ]:





# In[ ]:




