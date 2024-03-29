#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st


from scipy import stats
import requests
import io
from sklearn.metrics import accuracy_score

import html5lib


from datetime import date
from datetime import datetime

import math


# In[2]:


##Updating Page Logo and Tab Title
st.set_page_config(page_title='NFL Game Predictor',
                   page_icon='https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png',
                   layout="wide")



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
def accuracy(text):
     st.markdown(f'<p style="color:#013369;font-size:15px;border-radius:2%;">{text}</p>', unsafe_allow_html=True)
def disclaimer(text):
     st.markdown(f'<p style="color:red;font-size:15px;border-radius:2%;">{text}</p>', unsafe_allow_html=True)


# In[3]:


#Getting current week number

current_week_num = 1

season_start = datetime.strptime('2022-09-08', '%Y-%m-%d').date()

current_week_num = math.ceil(((date.today()-season_start).days/7)+.01)


# In[8]:



##--------------------------------------------------------Application Displayed Portion-----------------------------------------

#Reading modeldata from Github
preds_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/preds.csv'
preds = pd.read_csv(preds_url, error_bad_lines=False)

weather_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/weather_df.csv'
pic_home = 'https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png'
pic_vis = 'https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png'

df_weather =  pd.read_csv(weather_url, index_col=0)


##Header and Logo
col_title, col_logo = st.beta_columns([4,1])
with col_title:
  st.title('NFL Game Predictor')
  st.markdown(' ## Created by Cole Brandt')
  st.markdown('  Last updated: Saturday, January 21st, 2023')  
  accuracy('Lifetime ATS accuracy: 54.6%')
  #accuracy('Lifetime ML accuracy: 67.3%')
with col_logo:
  st.image("https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png")
st.write("#")
#highlight('NFL Week ' + str(current_week_num))
highlight('NFL Week: Divisional Playoffs')



        
visitor = ['']
home = ['']

# pic_home = 'https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png'
# pic_vis = 'https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png'

# c_visitor1, vp1, c_home1, cp1, c_weather1, c_spread1, c_prediction1, c_spread_beater1 = st.beta_columns([1,1,1,1,1,1,1,1])
# with c_visitor1:
#     st.markdown('Visitor')
# with vp1:
#     st.markdown('')
# with c_home1:
#     st.markdown('Home')
# with cp1:
#     st.markdown('')
# with c_weather1:
#     st.markdown('Weather')
# with c_spread1:
#     st.markdown('Spread')
# with c_prediction1:
#     st.markdown('Predicted Score Difference')
# with c_spread_beater1:
#     st.markdown('Winner Against the Spread')

# for i in range(0,len(preds['Visitor'])):
#     c_visitor, vp, c_home, cp, c_weather, c_spread, c_prediction, c_spread_beater = st.beta_columns([1,1,1,1,1,1,1,1])
#     with c_visitor:
#         st.markdown(preds['Visitor'][i])
#     with vp:
#         if 'Arizona' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/177/full/kwth8f1cfa2sch5xhjjfaof90.png'
#         elif 'Atlanta' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/173/full/299.png'
#         elif 'Baltimore' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/153/full/318.png'
#         elif 'Buffalo' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/149/full/n0fd1z6xmhigb0eej3323ebwq.png'   
#         elif 'Carolina' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/174/full/f1wggq2k8ql88fe33jzhw641u.png'    
#         elif 'Chicago' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/169/full/364.png'    
#         elif 'Cincinnati' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/154/full/cincinnati_bengals_logo_primary_2021_sportslogosnet-2049.png'
#         elif 'Cleveland' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/155/full/7855_cleveland_browns-primary-2015.png'
#         elif 'Dallas' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/165/full/406.png'
#         elif 'Denver' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/161/full/9ebzja2zfeigaziee8y605aqp.png'
#         elif 'Detroit' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/170/full/1398_detroit_lions-primary-2017.png'
#         elif 'Green Bay' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/171/full/dcy03myfhffbki5d7il3.png'
#         elif 'Houston' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/157/full/570.png'
#         elif 'Indianapolis' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/158/full/593.png'
#         elif 'Jacksonville' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/159/full/8856_jacksonville_jaguars-alternate-2013.png'   
#         elif 'Kansas City' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/162/full/857.png'    
#         elif 'Las Vegas' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/6708/full/8521_las_vegas_raiders-primary-20201.png'    
#         elif 'Chargers' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/6446/full/1660_los_angeles__chargers-primary-20201.png'
#         elif 'Rams' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/5941/full/8334_los_angeles_rams-primary-20201.png'
#         elif 'Miami' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/150/full/7306_miami_dolphins-primary-2018.png'
#         elif 'Minnesota' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/172/full/2704_minnesota_vikings-primary-20131.png'
#         elif 'New England' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/151/full/y71myf8mlwlk8lbgagh3fd5e0.png'
#         elif 'New Orleans' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/175/full/907.png'
#         elif 'Giants' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/166/full/80fw425vg3404shgkeonlmsgf.png'
#         elif 'Jets' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/152/full/9116_new_york_jets-primary-2019.png'
#         elif 'Philadelphia' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/167/full/960.png'   
#         elif 'Pittsburgh' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/156/full/970.png'    
#         elif 'San Francisco' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/179/full/9455_san_francisco_49ers-primary-2009.png'    
#         elif 'Seattle' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/180/full/pfiobtreaq7j0pzvadktsc6jv.png'
#         elif 'Tampa Bay' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/176/full/8363_tampa_bay_buccaneers-primary-2020.png'
#         elif 'Tennessee' in preds['Visitor'][i]:
#             pic_vis = 'https://content.sportslogos.net/logos/7/160/full/1053.png'
#         else:
#             pic_vis = 'https://content.sportslogos.net/logos/7/6741/full/8837_washington_football_team-wordmark-20201.png'
#         st.image(pic_vis)
#     with c_home:
#         st.markdown(preds['Home'][i])
#     with cp:
#         if 'Arizona' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/177/full/kwth8f1cfa2sch5xhjjfaof90.png'
#         elif 'Atlanta' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/173/full/299.png'
#         elif 'Baltimore' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/153/full/318.png'
#         elif 'Buffalo' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/149/full/n0fd1z6xmhigb0eej3323ebwq.png'   
#         elif 'Carolina' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/174/full/f1wggq2k8ql88fe33jzhw641u.png'    
#         elif 'Chicago' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/169/full/364.png'    
#         elif 'Cincinnati' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/154/full/cincinnati_bengals_logo_primary_2021_sportslogosnet-2049.png'
#         elif 'Cleveland' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/155/full/7855_cleveland_browns-primary-2015.png'
#         elif 'Dallas' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/165/full/406.png'
#         elif 'Denver' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/161/full/9ebzja2zfeigaziee8y605aqp.png'
#         elif 'Detroit' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/170/full/1398_detroit_lions-primary-2017.png'
#         elif 'Green Bay' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/171/full/dcy03myfhffbki5d7il3.png'
#         elif 'Houston' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/157/full/570.png'
#         elif 'Indianapolis' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/158/full/593.png'
#         elif 'Jacksonville' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/159/full/8856_jacksonville_jaguars-alternate-2013.png'   
#         elif 'Kansas City' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/162/full/857.png'    
#         elif 'Las Vegas' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/6708/full/8521_las_vegas_raiders-primary-20201.png'    
#         elif 'Chargers' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/6446/full/1660_los_angeles__chargers-primary-20201.png'
#         elif 'Rams' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/5941/full/8334_los_angeles_rams-primary-20201.png'
#         elif 'Miami' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/150/full/7306_miami_dolphins-primary-2018.png'
#         elif 'Minnesota' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/172/full/2704_minnesota_vikings-primary-20131.png'
#         elif 'New England' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/151/full/y71myf8mlwlk8lbgagh3fd5e0.png'
#         elif 'New Orleans' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/175/full/907.png'
#         elif 'Giants' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/166/full/80fw425vg3404shgkeonlmsgf.png'
#         elif 'Jets' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/152/full/9116_new_york_jets-primary-2019.png'
#         elif 'Philadelphia' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/167/full/960.png'   
#         elif 'Pittsburgh' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/156/full/970.png'    
#         elif 'San Francisco' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/179/full/9455_san_francisco_49ers-primary-2009.png'    
#         elif 'Seattle' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/180/full/pfiobtreaq7j0pzvadktsc6jv.png'
#         elif 'Tampa Bay' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/176/full/8363_tampa_bay_buccaneers-primary-2020.png'
#         elif 'Tennessee' in preds['Home'][i]:
#             pic_home = 'https://content.sportslogos.net/logos/7/160/full/1053.png'
#         else:
#             pic_home = 'https://content.sportslogos.net/logos/7/6741/full/8837_washington_football_team-wordmark-20201.png'
#         st.image(pic_home)
#     with c_weather:
#         home_full_c = ''
#         if preds['Home Full'][i] in df_weather['Home_Team']:
#             home_full_c = preds['Home Full'][i]
#             st.text('Weather: '  + str(df_weather[df_weather['Home_Team'] == home_full_c].iat[0,2]))
#             st.text('Temperature (degrees Fahrenheit): '  + str(df_weather[df_weather['Home_Team'] == home_full_c].iat[0,3]))
#             st.text('Wind (mph): ' + str(df_weather[df_weather['Home_Team'] == home_full_c].iat[0,4]))
#         else:
#             st.markdown('Weather for this game is not yet available.')
#     with c_spread:
#         st.markdown(preds['Team_Spread'][i])
#     with c_prediction:
#         st.markdown(preds['Predicted_Difference'][i])
#     with c_spread_beater:
#         st.markdown(preds['Spread_Beater'][i])
    


# In[9]:


# preds_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/preds.csv'
# preds = pd.read_csv(preds_url, error_bad_lines=False)
#preds


# In[10]:


#Predicted result:
preds['Model Prediction'] = ''

for p in range(0,len(preds['Predicted_Difference'])):
    preds['Model Prediction'][p] = np.where(preds['Predicted_Difference'][p] < 0, preds['Underdog'][p] + ' by ' + str(abs(round(preds['Predicted_Difference'][p], 1))),
                                    np.where(preds['Predicted_Difference'][p] > 0, preds['Favorite'][p] + ' by ' + str(abs(round(preds['Predicted_Difference'][p], 1))), 'PUSH' ))


# In[11]:


#df_weather[df_weather['Home_Team'] == 'Los Angeles Rams'].iat[0,2]
#len(preds['Predicted_Difference'])
#df_weather


# In[12]:


preds['Weather'] = ''


for p in range(0,len(preds['Predicted_Difference'])):
    try:
        preds['Weather'][p] = ('Weather: ' + df_weather[df_weather['Home_Team'] == preds['Home Full'][p]].iat[0,2] + '     ' + """        
                                Temperature (Degrees Fahrenheit): """ + str(df_weather[df_weather['Home_Team'] == preds['Home Full'][p]].iat[0,3]) + '     ' + """                              
                                Wind (mph): """ + str(df_weather[df_weather['Home_Team'] == preds['Home Full'][p]].iat[0,4]))
        if (len(preds['Weather'][p]) < 2):
            preds['Weather'][p] = 'Weather Not Available'
    except IndexError as err:
        preds['Weather'][p] = 'Weather Not Available'
        #print(err)
    
#preds['Weather']




# In[15]:


#preds['Weather']


# In[14]:


#Final displayed table

final = preds[['Visitor', 'Home', 'Game Time', 'Weather', 'Team_Spread', 'Model Prediction', 'Spread_Beater', 'Probability' ]]
final = final.rename(columns = {'Team_Spread': 'Spread', 'Spread_Beater': 'Winner Against Spread', 'Probability': 'Confidence'})

#Listing games by game index beginning at 1
final.index = final.index+1

#Final table display
st.table(final)


# In[45]:


#df_full.head()


# In[60]:


df_full_url = 'https://raw.githubusercontent.com/crbrandt/NFLPredictor/main/Data/df_full.csv'


df_full =  pd.read_csv(df_full_url, index_col=0)
df_full = df_full.sort_values(by=['adj_elo'], ascending = False)
df_full = df_full.reset_index(drop=True)
df_full.index = df_full.index+1

for i in range(0, len(df_full['Team_x'])):
    df_full.iat[i,4] = str(round(df_full.iat[i,4],1))
    df_full.iat[i,5] = str(round(df_full.iat[i,5],1).astype(str))
    df_full.iat[i,6] = str(round(df_full.iat[i,6],3).astype(str))
    df_full.iat[i,11] = str(round(df_full.iat[i,11],3).astype(str))
    df_full.iat[i,8] = str(round(df_full.iat[i,8],1).astype(str))
    df_full.iat[i,13] = str(round(df_full.iat[i,13],1).astype(str))
    df_full.iat[i,7] = str(round(df_full.iat[i,7],1).astype(str))
    df_full.iat[i,12] = str(round(df_full.iat[i,12],1).astype(str))
    df_full.iat[i,9] = str(round(df_full.iat[i,9],1).astype(str))
    df_full.iat[i,14] = str(round(df_full.iat[i,14],1).astype(str))
    df_full.iat[i,10] = str(round(df_full.iat[i,10],2).astype(str))
    

df_full['PFpg'] = df_full['PFpg'].astype(str)
df_full['PApg'] = df_full['PApg'].astype(str)
df_full['CompPCT_Off'] = df_full['CompPCT_Off'].astype(str)
df_full['PassYardspg'] = df_full['PassYardspg'].astype(str)
df_full['RushYardspg'] = df_full['RushYardspg'].astype(str)
df_full['OffSackspg'] = df_full['OffSackspg'].astype(str)
df_full['TurnoverMargin'] = df_full['TurnoverMargin'].astype(str)
df_full['CompPCT_Def'] = df_full['CompPCT_Def'].astype(str)
df_full['PassYardspg_Def'] = df_full['PassYardspg_Def'].astype(str)
df_full['RushYardspg_Def'] = df_full['RushYardspg_Def'].astype(str)
df_full['DefSackspg'] = df_full['DefSackspg'].astype(str)

df_full = df_full.rename(columns = {'Team_x': 'Team Name', 'Team_y': 'Nickname', 'adj_elo': 'ELO Rating'})

#df_full

disclaimer('Note: This model does not currently account for players being out due to COVID or injury.')

# In[50]:


st.markdown('___')
Rankings = st.beta_expander('Statistics and Power Rankings')
with Rankings:
    st.markdown('Team Statistics Sorted by Power Ranking')
    df_full


# In[31]:


# Bottom info bar ------------------------------------------------------------------------
st.markdown('___')
about = st.beta_expander('About')
with about:
    '''
    Thank you for visiting the NFL Game Predictor, developed by Cole Brandt. For more information, please visit my [Github repository] (https://github.com/crbrandt/NFLPredictor).
    
    Feel free to support via Venmo, @ColeBrandt
    
    Spreads from MGM Sportsbook, scraped from VegasInsider.com. All images sourced from sportslogos.net. 
    
    [Contact Me] (mailto:cole.r.brandt@gmail.com)
    '''
    
st.image("https://static.wikia.nocookie.net/logopedia/images/b/bc/NationalFootballLeague_PMK01a_1940-1959_SCC_SRGB.png",
    width= 100, caption='2021 Cole Brandt')


# In[ ]:




