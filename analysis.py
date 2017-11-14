# Peri Akiva, Arpit Shah

import json
import sys
from textblob import TextBlob

# scrap from website for comparison
def getApprovalRatings(jsonFile):
    pass

def impactForText(string):
    text = TextBlob(string)
    polarity = text.sentiment.polarity
    if polarity>0:
        return polarity*100
    if polarity<0:
        return polarity*100
    return 0
 
def impactForList(tweets):
    # overAll is an approval index ranging from 0-100
    overAll = 0
    for i in tweets:
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


tpd = {'day1':["i hate sushi","i love europe","trump is not good for america","where is my phone?"],'day2':["i hate trump","trump is the best","hello there"]}
print impactForData(tpd)
#tw = ["i hate sushi","i love europe","trump is not good for america","where is my phone?"]
#print impactForList(tw)
