# Peri Akiva, Arpit Shah
import re
import pickle as p
import peakutils
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import json
import numpy as np
import sys
from textblob import TextBlob


"""
def DFtoDict(df):
    trumpTweets={}
    for index,row in df.iterrows():
        print str(row['enddate'])
        trumpTweets[str(row['enddate'])]=trumpTweets[str(df['enddate'])]+[str(row[''])]
    return trumpTweets
"""
#dfTrump = csvToDataFrame('/home/native/projects/semanticIMpactAnalysis/tweets_from_jan20_dec02.csv')

def DFDuplicateHandle(df):
    df = df.groupby('enddate',as_index=False)['adjusted_approve','adjusted_disapprove'].mean()
    df['enddate'] = pd.to_datetime(df['enddate'])
    df = df.sort_values(by='enddate')
    #print df
    return df

def csvToDataFrame(pathToCsv):
    return pd.read_csv(pathToCsv,usecols=['startdate','enddate','pollster','adjusted_approve','adjusted_disapprove'])

df = csvToDataFrame('/home/native/projects/semanticImpactAnalysis/approval_polllist.csv')
df = DFDuplicateHandle(df)
#df.to_csv('cleanPollData.csv')
#print DFtoDict(df)
def plotDF(df):
    #df['enddate'] = pd.to_datetime(df['enddate'])
    fig = plt.figure()
    plot = fig.add_subplot(111)

    def on_plot_hover(event):

        #x= event.xdata
        print event



    plot.plot_date(df['enddate'],df['adjusted_approve'],'b-',linewidth=0.4)
    plot.plot_date(df['enddate'],df['adjusted_disapprove'],'r-',linewidth=0.4)
    plot.grid(color='black',linestyle='-',linewidth=0.1)
    plt.xlabel('Date')
    plt.ylabel('Approval/Disapproval Rate')
    plt.ylim(0,100,5)
    plt.title('Trumps Approval Rating Compared to Twitter Users Sentiments')
    annot = plt.annotate("",xy=(0,0),xytext=(20,20),textcoords="offset points",bbox=dict(boxstyle="round",fc="w"),arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect('motion_notify_event',on_plot_hover)
    plt.show()
#plotDF(df)


def impactForText(string):
    text = TextBlob(string)
    polarity = text.sentiment.polarity
    if polarity>0:
        return polarity
    if polarity<0:
        return polarity
    return 0
print impactForText('i hate sushi') 
print impactForText('evil') 
print impactForText('i hate sushi') 
print impactForText('i hate sushi') 
print impactForText('i hate sushi') 
def impactForList(tweets):
    # overAll is an approval index ranging from 0-100
    overAll = 0
    for i in tweets:
#        print impactForText(i)
        overAll+=impactForText(i)
    # normalize the data
    overAll=(overAll+100)/2
    return overAll

def impactForData(dic):
    impactDict = {}
    for i in dic:
        impactDict[i] = impactForList(dic[i])
    # returns approval index per day of tweets
    return impactDict

trumpTweets = p.load(open("trumps_tweets_dict.p","rb"))
#x= impactForData(trumpTweets)
#print x.values()
#print np.mean(x.values())
tdp = {'day1':['i hate sushi','work with me here','trump is evil','why are you killing me'],'day2':['evil hate dislike shit no bad']}
#print impactForData(tdp)
#print impactForData(tpd)
#tw = ["i hate sushi","i love europe","trump is not good for america","where is my phone?"]
#print impactForList(tw)
