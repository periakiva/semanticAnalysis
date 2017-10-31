import sys
from textblob import TextBlob

def impactForText(string):
    text = TextBlob(string)
    polarity = text.sentiment.polarity
    if polarity>0.2:
        return polarity*100
    if polarity<-0.2:
        return polarity*100
    else:
        return 0
 
def impactForList(tweets):
    overAll = 0
    for i in tweets:
        overAll+=impactForText(i)
    overAll=(overAll+100)/2
    return overAll

def impactForData(dic):
    impactDict = {}
    for i in dic:
        


tw = ["i hate sushi","i love europe","trump is not good for america","where is my phone?"]
print impactForList(tw)
