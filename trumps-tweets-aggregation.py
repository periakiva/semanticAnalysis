
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


ranges = pd.read_csv('./cleanPollData.csv')
end_dates = ranges['enddate']


# In[30]:


trump_tweets = pd.read_csv('./tweets_from_jan20_dec02.csv')
# trump_tweets['text'][0]


# In[35]:


filtered_data = trump_tweets[['text', 'timestamp']]


# In[36]:


filtered_data


# In[46]:


from datetime import datetime
# datetime_object = datetime.strptime('2017-01-22', '%Y-%m-%d')


# In[50]:


result_dict = { }
tmp_list = []
last_date = None
for index, row in filtered_data.iterrows():
    dt = datetime.strptime(row['timestamp'][:10], '%Y-%m-%d')
    text = row['text']
    if last_date == dt:
        tmp_list.append(text)
    else:
        if last_date:
            result_dict[last_date.isoformat()[:10]] = tmp_list
        tmp_list = []
        last_date = dt
        tmp_list.append(text)
result_dict


# In[51]:


import pickle
pickle.dump( result_dict, open( "trumps_tweets_dict.p", "wb" ) )


# In[17]:


datetime_object

