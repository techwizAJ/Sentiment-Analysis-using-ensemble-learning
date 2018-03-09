from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from sentiment_mod import sentiment


#consumer key, consumer secret, access token, access secret write your own code
ckey=""
csecret=""
atoken=""
asecret=""

class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = all_data["text"]
            sentiment_value, confidence = sentiment(tweet)
            if ( sentiment_value > 0):
                sentiment_express = "pos"
            elif (sentiment_value <= 0 ):
                sentiment_express ="neg"
            print(tweet, sentiment_express, confidence)

            if confidence*100 >= 80:
                output = open("twitter-out.txt","a")
                output.write(sentiment_express)
                output.write('\n')
                output.close()

            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
print(" \t \t \t  \t Subject Sentiment Analysis  Tool  -  Arihant Jain  \t \t \t \t ")
print("\n This Tool crawls over Twitter and analyze tweets of people based on the subject and gives out polarity of the sentiment based on the text  \n ")
print(" \t Enter Subject Title: ")
tracking = input(" ------>>\n")
twitterStream.filter(track=[tracking])
