#!/usr/bin/env python
# coding: utf-8

# In[234]:


import urllib
import urllib.request
import urllib.parse

from bs4 import BeautifulSoup
import requests
#pip install html5lib
import html5lib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from dateutil import parser
from decimal import *
import re


# In[291]:


driver = webdriver.Chrome('/Users/colebrandt/Downloads/chromedriver')


# In[292]:


### ELO

driver.get('https://projects.fivethirtyeight.com/2021-nfl-predictions/')
elo_teams = driver.find_elements_by_xpath('//td[@class="team"]')

elo_teams_list = []
for t in range(len(elo_teams)):
    elo_teams_list.append(elo_teams[t].text)


elo_teams_list2 = []

for t in elo_teams_list:
    team_str = ''
    for tt in t:
        if tt.isalpha():
            team_str += tt
    if team_str == 'ers':
        team_str = '49ers'
    elo_teams_list2.append(team_str)


# In[293]:


elo_page = requests.get("https://projects.fivethirtyeight.com/2021-nfl-predictions/")

soup = BeautifulSoup(elo_page.content, 'html.parser')

#print(soup.prettify())
#list(soup.children)
#[type(item) for item in list(soup.children)]
html = list(soup.children)[1]

#list(html.children)
body = list(html.children)[1]
p = list(body.children)[3]
#list(body.children)

#list(html.children)
#p = list(body.children)[1]
#body = list(html.children)[3]
#p
pp = list(p.children)[1]
ppp = list(pp.children)[2]
pppp = list(ppp.children)[0]
ppppp = list(pppp.children)[0]
pt = ppppp.get_text()


# In[207]:


Index = pt.index('super bowl')
pt2 = pt[Index + 10:]
pt2 = pt2.replace('49ers', 'FortyNiners')

#pt2
pt3 = pt2.split('%')

pt4 = []
pt5_all = []


for item in pt3:
    if len(item) >= 5:
        pt4.append(int(item[:4]))
        pt5_all.append(item)

        
pt_qb_adj = []
sub = ''

for item in pt5_all:
    if ('-' in item[5: 9]):
        #pt_qb_adj.append(item[4:10][item[4:10].find('-')+1, item[4:10].find('-')+3])
        sub = item[5:10]
        dash = sub.find('-')
        pt_qb_adj.append(-1*int(re.sub("[^0-9]", "", sub[dash+1:dash+3])))
    elif('+' in item[5:9]):
        #pt_qb_adj.append(item[4:10][item[4:10].find('+')+1, item[4:10].find('+')+3])
        sub = item[5:10]
        dash = sub.find('+')
        pt_qb_adj.append(int(re.sub("[^0-9]", "", sub[dash+1:dash+3])))
    else:
        pt_qb_adj.append(0)
        
        
    
        
#pt4

elos_df = pd.DataFrame({'Team':elo_teams_list2,'elo':pt4, 'qb_adj': pt_qb_adj})
#elos_df
#pt5_all

elos_df['adj_elo'] = elos_df['elo'] + elos_df['qb_adj']


##########################################################elos_df


# In[208]:


##
##Pulling in Offensive Data
##

offensive_url = 'https://www.pro-football-reference.com/years/2021/'
defensive_url = 'https://www.pro-football-reference.com/years/2021/opp.htm'
    
driver.get(offensive_url)


# rowCount = driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr')
# len(rowCount)

nfl_list = []
pf_list = []
pa_list = []


for row in [2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20]:
    nfl_list.append(((driver.find_elements_by_xpath('//*[@id="AFC"]/tbody/tr[' + str(row) + ']/th'))[0].text))
    nfl_list.append(((driver.find_elements_by_xpath('//*[@id="NFC"]/tbody/tr[' + str(row) + ']/th'))[0].text))
    pf_list.append(((driver.find_elements_by_xpath('//*[@id="AFC"]/tbody/tr[' + str(row) + ']/td[4]'))[0].text))
    pf_list.append(((driver.find_elements_by_xpath('//*[@id="NFC"]/tbody/tr[' + str(row) + ']/td[4]'))[0].text))
    pa_list.append(((driver.find_elements_by_xpath('//*[@id="AFC"]/tbody/tr[' + str(row) + ']/td[5]'))[0].text))
    pa_list.append(((driver.find_elements_by_xpath('//*[@id="NFC"]/tbody/tr[' + str(row) + ']/td[5]'))[0].text))
    
nfl_record_df = pd.DataFrame(zip(nfl_list, pf_list, pa_list))
nfl_record_df.rename(columns={0:'Team', 1:'PF', 2:'PA'}, inplace=True)


nfl_off_teams = []
nfl_g = []
nfl_off_comp = []
nfl_off_pass_att = []
nfl_off_pass_yds = []
nfl_off_rush_yds = []
nfl_team_TO = []


rl = [*range(1,21), *range(23,35)]
#

for row in rl:
    nfl_off_teams.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[1]/a'))[0].text))  
    nfl_g.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[2]'))[0].text))
    nfl_off_comp.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[10]'))[0].text))
    nfl_off_pass_att.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[11]'))[0].text))
    nfl_off_pass_yds.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[12]'))[0].text))
    nfl_off_rush_yds.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[18]'))[0].text))
    nfl_team_TO.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[7]'))[0].text))
    
nfl_off_df = pd.DataFrame(zip(nfl_off_teams, nfl_g, nfl_off_comp, nfl_off_pass_att, nfl_off_pass_yds, nfl_off_rush_yds, nfl_team_TO))

nfl_off_df.rename(columns={0:'Team', 1:'G', 2:'Passes Completed', 3:'Pass Attempts', 4:'Pass Yards', 5:'Rush Yards', 6:'Turnovers'}, inplace=True)


#nfl_off_df


# In[209]:


nfl_sack_teams = []
nfl_sacks_off = []

driver.get(offensive_url)

rp = [*range(1,21), *range(22,34)]


for row in rp:
    nfl_sack_teams.append(((driver.find_elements_by_xpath('//*[@id="passing"]/tbody/tr[' + str(row) + ']/td[1]'))[0].text))  
    nfl_sacks_off.append(((driver.find_elements_by_xpath('//*[@id="passing"]/tbody/tr[' + str(row) + ']/td[17]'))[0].text))  
    

off_sack_df = pd.DataFrame(zip(nfl_sack_teams, nfl_sacks_off))
off_sack_df = off_sack_df.rename(columns={0:'Team', 1:'Off_Sacks'})

pd.options.display.max_rows = 50
#//*[@id="passing"]/tbody/tr[20]/td[1]/a
#//*[@id="passing"]/tbody/tr[20]/td[1]/a


# In[37]:


off_sack_df


# In[210]:


nfl_def_teams = []
nfl_def_comp = []
nfl_def_pass_att = []
nfl_def_pass_yds = []
nfl_def_rush_yds = []
nfl_def_TO = []
nfl_def_sacks = []

driver.get(defensive_url)

rd = [*range(1,21), *range(23,35)]


for row in rd:
    nfl_def_teams.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[1]'))[0].text))  
    nfl_def_comp.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[10]'))[0].text))
    nfl_def_pass_att.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[11]'))[0].text))
    nfl_def_pass_yds.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[12]'))[0].text))
    nfl_def_rush_yds.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[18]'))[0].text))
    nfl_def_TO.append(((driver.find_elements_by_xpath('//*[@id="team_stats"]/tbody/tr[' + str(row) + ']/td[7]'))[0].text))
    
nfl_def_df = pd.DataFrame(zip(nfl_def_teams, nfl_def_comp, nfl_def_pass_att, nfl_def_pass_yds, nfl_def_rush_yds, nfl_def_TO))
nfl_def_df.rename(columns={0:'Team',  1:'Passes_Completed_Def', 2:'Pass_Attempts_Def', 3:'Pass_Yards_Def', 4:'Rush_Yards_Def', 5:'Turnovers_Def'}, inplace=True)

#nfl_def_df


# In[211]:


nfl_sack_teams_def = []
nfl_sacks_def = []

driver.get(defensive_url)

rp = [*range(1,21), *range(22,34)]


for row in rp:
    nfl_sack_teams_def.append(((driver.find_elements_by_xpath('//*[@id="passing"]/tbody/tr[' + str(row) + ']/td[1]'))[0].text))  
    nfl_sacks_def.append(((driver.find_elements_by_xpath('//*[@id="passing"]/tbody/tr[' + str(row) + ']/td[17]'))[0].text))  
    

def_sack_df = pd.DataFrame(zip(nfl_sack_teams_def, nfl_sacks_def))
def_sack_df.rename(columns={0:'Team', 1:'Def_Sacks'})


# In[212]:


current_df = nfl_record_df.merge(nfl_off_df, left_on='Team', right_on='Team').merge(off_sack_df, left_on='Team', right_on='Team').merge(nfl_def_df, left_on='Team', right_on = 'Team').merge(def_sack_df, left_on='Team', right_on = 0)
current_df = current_df.rename(columns={1:'Def_Sacks'})
current_df = current_df.drop(columns = 0)


#current_df = current_df.merge()
elos_df['Full Team Name'] = np.select(
    [
        elos_df['Team'] == 'Bills', 
        elos_df['Team'] == 'Buccaneers',
        elos_df['Team'] == 'Ravens', 
        elos_df['Team'] == 'Chargers',
        elos_df['Team'] == 'Rams', 
        elos_df['Team'] == 'Cardinals',
        elos_df['Team'] == 'Cowboys', 
        elos_df['Team'] == 'Packers',
        elos_df['Team'] == 'Chiefs', 
        elos_df['Team'] == 'Browns',
        elos_df['Team'] == 'Saints', 
        elos_df['Team'] == 'Titans',
        elos_df['Team'] == 'Seahawks', 
        elos_df['Team'] == '49ers',
        elos_df['Team'] == 'Broncos', 
        elos_df['Team'] == 'Vikings',
        elos_df['Team'] == 'Patriots', 
        elos_df['Team'] == 'Colts',
        elos_df['Team'] == 'Bengals', 
        elos_df['Team'] == 'Panthers',
        elos_df['Team'] == 'Raiders', 
        elos_df['Team'] == 'Washington',
        elos_df['Team'] == 'Bears', 
        elos_df['Team'] == 'Steelers',
        elos_df['Team'] == 'Eagles', 
        elos_df['Team'] == 'Falcons',
        elos_df['Team'] == 'Giants', 
        elos_df['Team'] == 'Dolphins',
        elos_df['Team'] == 'Jets', 
        elos_df['Team'] == 'Jaguars',
        elos_df['Team'] == 'Lions', 
        elos_df['Team'] == 'Texans'  
    ], 
    [
        'Buffalo Bills', 
        'Tampa Bay Buccaneers',
        'Baltimore Ravens', 
        'Los Angeles Chargers',
        'Los Angeles Rams', 
        'Arizona Cardinals',
        'Dallas Cowboys', 
        'Green Bay Packers',
        'Kansas City Chiefs', 
        'Cleveland Browns',
        'New Orleans Saints', 
        'Tennessee Titans',
        'Seattle Seahawks', 
        'San Francisco 49ers',
        'Denver Broncos', 
        'Minnesota Vikings',
        'New England Patriots', 
        'Indianapolis Colts',
        'Cincinnati Bengals', 
        'Carolina Panthers',
        'Las Vegas Raiders', 
        'Washington Football Team',
        'Chicago Bears', 
        'Pittsburgh Steelers',
        'Philadelphia Eagles', 
        'Atlanta Falcons',
        'New York Giants', 
        'Miami Dolphins',
        'New York Jets', 
        'Jacksonville Jaguars',
        'Detroit Lions', 
        'Houston Texans'
    ], 
    default='Unknown'
)

current_df = current_df.merge(elos_df, left_on='Team', right_on = 'Full Team Name')

current_df = current_df.drop(columns = ['Full Team Name', 'elo', 'qb_adj'])

#current_df

current_df['PFpg'] = current_df['PF'].astype(int)/current_df['G'].astype(int)
current_df['PApg'] = current_df['PA'].astype(int)/current_df['G'].astype(int)
current_df['CompPCT_Off'] = current_df['Passes Completed'].astype(int)/current_df['Pass Attempts'].astype(int)
current_df['PassYardspg'] = current_df['Pass Yards'].astype(int)/current_df['G'].astype(int)
current_df['RushYardspg'] = current_df['Rush Yards'].astype(int)/current_df['G'].astype(int)
current_df['OffSackspg'] = current_df['Off_Sacks'].astype(int)/current_df['G'].astype(int)
current_df['TurnoverMargin'] = current_df['Turnovers_Def'].astype(int)/current_df['Turnovers'].astype(int)

current_df['CompPCT_Def'] = current_df['Passes_Completed_Def'].astype(int)/current_df['Pass_Attempts_Def'].astype(int)
current_df['PassYardspg_Def'] = current_df['Pass_Yards_Def'].astype(int)/current_df['G'].astype(int)
current_df['RushYardspg_Def'] = current_df['Rush_Yards_Def'].astype(int)/current_df['G'].astype(int)
current_df['DefSackspg'] = current_df['Def_Sacks'].astype(int)/current_df['G'].astype(int)



current_df_full = current_df.drop(columns = ['PF', 'PA', 'Passes Completed', 'Pass Attempts', 'Pass Yards', 'Rush Yards', 'Turnovers', 'Off_Sacks', 
                                       'Passes_Completed_Def', 'Pass_Attempts_Def', 'Pass_Yards_Def', 'Rush_Yards_Def', 'Turnovers_Def', 'Def_Sacks'])


current_df_full.to_csv("/Users/colebrandt/Documents/NFL_Predictor/Data/df_full.csv")


current_df_full



# In[55]:





# In[213]:


##
##Pulling in Weather Data
##
driver.get('https://www.vegasinsider.com/nfl/weather/')

(driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr[2]/td[1]/b[2]/a'))[0].text

home_team_list =[]
away_team_list = []
weather_list = []
wind_list = []
temp_list = []

rowCount = driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr')
len(rowCount)

for row in range(2, len(rowCount)):
    home_team_list.append((driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr[' + str(row) + ']/td[1]/b[2]/a'))[0].text)
    away_team_list.append((driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr[' + str(row) + ']/td[1]/b[1]/a'))[0].text)
    weather_list.append((driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr[' + str(row) + ']/td[3]'))[0].text)
    wind_list.append((driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr[' + str(row) + ']/td[4]'))[0].text)
    temp_list.append((driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table[2]/tbody/tr[2]/td/table/tbody/tr[' + str(row) + ']/td[5]'))[0].text)


weather_df = pd.DataFrame(zip(away_team_list, home_team_list, weather_list, wind_list, temp_list))
weather_df.rename(columns={0:'Away_Team', 1:'Home_Team', 2:'Weather', 3:'Wind', 4:'Temp'}, inplace=True)

weather_df['Weather'] = np.where(weather_df['Weather'] == ' ', 'DOME', weather_df['Weather'])
weather_df['Wind'] = np.where(weather_df['Wind'] == ' ', '0-0', weather_df['Wind'])
weather_df['Temp'] = np.where(weather_df['Temp'] == ' ', 72, weather_df['Temp'])

for row in range(0,weather_df.shape[0]):
     weather_df.loc[row, 'Wind'] = re.sub('[^1234567890-]', '', 
                                 weather_df.loc[row, 'Wind'])

weather_df[['Wind_Min','Wind_Max']] = weather_df['Wind'].str.split('-',expand=True)
weather_df['Game_Wind_Avg'] = (weather_df['Wind_Min'].astype(int)+weather_df['Wind_Max'].astype(int))/2

# weather_df.shape[0]

weather_df = weather_df.drop(columns = ['Wind', 'Wind_Min', 'Wind_Max'])

weather_df.to_csv('/Users/colebrandt/Documents/NFL_Predictor/Data/weather_df.csv')


# In[256]:


driver.get('https://www.vegasinsider.com/nfl/odds/las-vegas/')

games = len(driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr'))
vteams = []
hteams = []
spread_list = []
gt_list = []

for row in range(1,games):
    try: 
        vteams.append(driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[' + str(row) + ']/td[1]/b[1]/a')[0].text)
        hteams.append(driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[' + str(row) + ']/td[1]/b[2]/a')[0].text)
        spread_list.append(driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[' + str(row) + ']/td[3]/a')[0].text)
        gt_list.append(driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[' + str(row) + ']/td[1]/span')[0].text)
    except IndexError as err:
        #print(err)
        break
    
    
# #driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table')
#//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[1]/td[3]/a/text()[3]
#//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[2]/td[3]/a/text()[3]

#driver.find_elements_by_xpath('//*[@id="wrapper"]/table/tbody/tr/td[2]/table/tbody/tr/td/div/table[2]/tbody/tr[1]/td[3]/a')[0].text

spread_list2=[]

for i in range(0,len(spread_list)):
    spread_list2.append(spread_list[i].split('\n')[0])
    spread_list2.append(spread_list[i].split('\n')[1])
    spread_list2.append(spread_list[i].split('\n')[2])
    
spread_list_f = []

for it in spread_list2:
    if (('u' not in it) & (it != ' ')):
        spread_list_f.append(it[:3].strip())   
        
sl = [] 
for it2 in spread_list_f:
    if (ord(it2[len(it2)-1:]) == 189):
        sl.append(it2[:len(it2)-1]+ '.5')
    else:
        sl.append(it2)
        
favlist = []
for it3 in spread_list:
    if 'u' in it3[0:10]:
        favlist.append('Home')
    else:
        favlist.append('Away')
    

fl = zip(vteams, hteams, sl, favlist, gt_list)
        
        


# In[266]:



fl_df = pd.DataFrame(fl)
fl_df


# In[267]:


len(fl_df[4])


# In[286]:


fl_df['DateTime'] = fl_df[4]
for d in range(0,len(fl_df[4])):
    fl_df['DateTime'][d] = (parser.parse(fl_df['DateTime'][d]) - timedelta(hours=3)).strftime("%m/%d %I:%M %p")
    
fl_df = fl_df.drop(columns = 4)


# In[287]:


fl_df = fl_df.rename(columns = {0:'Away',1:'Home',2:'Spread',3:'Favorite', 'DateTime': 'GameTime'})


# In[288]:


fl_df['fav_team'] = np.where(fl_df['Favorite']== 'Home', fl_df['Home'], fl_df['Away'])
fl_df['und_team'] = np.where(fl_df['Favorite']!= 'Home', fl_df['Home'], fl_df['Away'])

fl_df['fav_spread'] = fl_df['fav_team'] + ' ' + fl_df['Spread']


# In[289]:


fl_df.to_csv('/Users/colebrandt/Documents/NFL_Predictor/Data/spread_df.csv')


# In[290]:


fl_df


# In[ ]:




