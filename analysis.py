# Peri Akiva, Arpit Shah
import os
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
#from spacy.en import English
import spacy
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
import string

parser = spacy.load('en')
punctuations = string.punctuation
class predictors(TransformerMixin):
    def transform(self,X,**transform_params):
        return [clean_text(text) for text in X]
    def fit(self,X,y=None,**fit_params):
        return self
    def get_params(self,deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_!="-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer,ngram_range=(1,1))
classifier=LinearSVC()
pipe = Pipeline([("cleaner",predictors()),('vectorizer',vectorizer),('classifier',classifier)])

def csvToListTuples(csvPath):
    df = pd.read_csv(csvPath,usecols=['Sentiment','SentimentText'])
    labeledData=[(row['SentimentText'],row['Sentiment']) for index,row in df.iterrows()]
    with open('labeledData.pkl','wb') as f:
        p.dump(labeledData,f)
    return labeledData

if os.path.exists('labeledData.pkl'):
    with open('labeledData.pkl','rb') as f:
        labeledData = p.load(f)
else:
    labeledData = csvToListTuples('/home/native/projects/semanticImpactAnalysis/SemAnLabeled.csv')
#print labeledData

if os.path.getsize('labeledData.pkl')<10:
    os.remove('labeledData.pkl')
def splitData(labeledData):
    trainLabeled=[]
    testLabeled=[]
    n=0
    for i in labeledData:
        if n%10==0:
            testLabeled.append(i)
        else:
            trainLabeled.append(i)
        n=n+1
    return trainLabeled,testLabeled

train,test = splitData(labeledData)
#print train[:10]
#print test[:10]
#print len(train)
#print len(test)
pipe.fit([x[0] for x in train],[x[1] for x in train])
pred_data = pip.predict([x[0] for x in test])
for (sample,pred) in zip(test,pred_data):
    print sample,pred
print "Accuracy: ", accuracy_score([x[1] for x in test], pred_data)

#print labeledData

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

#df = csvToDataFrame('/home/native/projects/semanticImpactAnalysis/approval_polllist.csv')
#df = DFDuplicateHandle(df)
#df.to_csv('cleanPollData.csv')
#print DFtoDict(df)
def plotDF(df,x,y):
    #df['enddate'] = pd.to_datetime(df['enddate'])
    fig = plt.figure()
    plot = fig.add_subplot(111)

    def movingAvg(y):
        ynew=[50]
        ySum=50
        for i,k in zip(range(1,len(y)),y):
            ySum+=y[i]
            ynew.append(ySum/(i+1))
        return ynew

    ynew=movingAvg(y)

    def on_plot_hover(event):
        #x= event.xdata
        print event
    plot.plot_date(x,ynew,'k--',linewidth=0.9,label='Moving Average of Trump Tweets')
    plot.plot_date(x,y,'y-',linewidth=0.5,label='Trump Tweets')
    plot.plot_date(df['enddate'],df['adjusted_approve'],'b-',linewidth=0.4,label='Adjusted Approval')
    plot.plot_date(df['enddate'],df['adjusted_disapprove'],'r-',linewidth=0.4,label='Adjusted Disapproval')
    plot.grid(color='black',linestyle='-',linewidth=0.1)
    plt.legend(bbox_to_anchor=(0.,1.02,1.,-1.85),loc=10,ncol=2,borderaxespad=0.)
    plt.xlabel('Date')
    plt.ylabel('Approval/Disapproval Rate')
    plt.ylim(0,100,5)
    plt.title('Trumps Approval Rating Compared to Twitter Users Sentiments')
    annot = plt.annotate("",xy=(0,0),xytext=(20,20),textcoords="offset points",bbox=dict(boxstyle="round",fc="w"),arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect('motion_notify_event',on_plot_hover)
    plt.show()



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
def plotDF(df,x,y):
    #df['enddate'] = pd.to_datetime(df['enddate'])
    fig = plt.figure()
    plot = fig.add_subplot(111)

    def movingAvg(y):
        ynew=[50]
        ySum=50
        for i,k in zip(range(1,len(y)),y):
            ySum+=y[i]
            ynew.append(ySum/(i+1))
        return ynew

    ynew=movingAvg(y)

    def on_plot_hover(event):
        #x= event.xdata
        print event
    plot.plot_date(x,ynew,'r-',linewidth=0.5)
    plot.plot_date(x,y,'y-',linewidth=0.5)
    #plot.plot_date(dic.keys(),dic.values(),'y-',linewidth=0.5)
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


def impactForText(string):
    text = TextBlob(string)
    polarity = text.sentiment.polarity
    if polarity!=0:
        return polarity*len(re.sub("[^\w]"," ",string).split())*100
    return 0

print impactForText('i hate sushi')
print impactForText('evil')
print impactForText('bad')
print impactForText('bad')
print impactForText('evil bad bad')

def impactForList(tweets):
    # overAll is an approval index ranging from 0-100

    overAll = 0
    for i in tweets:
        #print impactForText(i)
        overAll+=impactForText(i)
    #normalize the data
    #overAll=(overAll+100)/2
    if overAll<=-2000:
        return -2000
    elif overAll>=2000:
        return 2000
    else:
        return overAll

def impactForData(dic):
    impactDict = {}
    for i in dic:
        impactDict[i] = impactForList(dic[i])
    # returns approval index per day of tweets
    return impactDict

trumpTweets = p.load(open("trumps_tweets_dict.p","rb"))
x= impactForData(trumpTweets)
keys = sorted(x.iterkeys())
maxi = max(x.values())
mini = min(x.values())
for key in sorted(x.keys()):
    print "%s : %s" % (key,x[key])
xnew={}
for key in sorted(x.keys()):
    xnew[key]=x[key]+(-mini)
    xnew[key]=int((xnew[key]/(-mini))*100)/2
#for key in sorted(xnew.keys()):
#    print "%s : %s" % (key,xnew[key])
print xnew
print min(xnew.values())
plotDF(df,sorted(xnew.keys()),[xnew[key] for key in sorted(xnew.keys())])
#print np.mean(x.values())
#tdp = {'day1':['i hate sushi','work with me here','trump is evil','why are you killing me'],'day2':['evil hate dislike shit no bad']}
#print impactForData(tdp)
#print impactForData(tpd)
#tw = ["i hate sushi","i love europe","trump is not good for america","where is my phone?"]
#print impactForList(tw)
