#Building voting system system on feature set to increase accuracy and realiablity of the classifier
#Building our own classifier by compling varoius classifiers based on voting of various classifiers

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
#wrapper of Sklearn Around nltk classifiers
from nltk.classify.scikitlearn import SklearnClassifier

#importing various sckitlearn clsssifiers and testing them on a training data
from sklearn.naive_bayes import MultinomialNB ,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):

	def __init__(self,*classifiers):
		self.classifiers = classifiers

	def classify(self,features):
		votes=[]
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)

		return mode(votes)

	def confidence(self,features):
		votes=[]
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes/len(votes)
		return conf





documents = []

for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		documents.append((list(movie_reviews.words(fileid)),category))

#random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

all_word = nltk.FreqDist(all_words)
word_features = list(all_word.keys())[:3000]

def find_features(document): #creating dictinary of word list in document tuple with boolean as a value
	words= set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words) 
	return features

#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = [(find_features(rev),category) for (rev,category)  in documents] 
# create a feature set with all words in document with category

traning_set = featuresets[100:]
testing_set = featuresets[:100]

classfier = nltk.NaiveBayesClassifier.train(traning_set) #using naive bayes algo to classify pos or neg movie reviews

#classfier_f = open("NaiveBayesSentiment.pickle","rb") #loading the trained model using pickle
#classfier = pickle.load(classfier_f)
#classfier_f.close()

print("ORiginal Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(classfier,testing_set))*100) #calculating accuracy of th model
classfier.show_most_informative_features(30)

MNB_classfier = SklearnClassifier(MultinomialNB())
MNB_classfier.train(traning_set)

print("Multinomial Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(MNB_classfier,testing_set))*100) #calculating accuracy of th model



B_classfier = SklearnClassifier(BernoulliNB())
B_classfier.train(traning_set)

print("Bernoulli Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(B_classfier,testing_set))*100) #calculating accuracy of th model



#LogisticRegression,SGDClassifier
#SVC,LinearSVC,NuSVC

LogisticRegression_classfier = SklearnClassifier(LogisticRegression())
LogisticRegression_classfier.train(traning_set)

print("LogisticRegression Algorithm Accuracy Percent : ",(nltk.classify.accuracy(LogisticRegression_classfier,testing_set))*100)

SGDClassifier_classfier = SklearnClassifier(SGDClassifier())
SGDClassifier_classfier.train(traning_set)

print("SGDClassifier_classfier  Algorithm Accuracy Percent : ",(nltk.classify.accuracy(SGDClassifier_classfier,testing_set))*100)

LinearSVC_classfier = SklearnClassifier(LinearSVC())
LinearSVC_classfier.train(traning_set)
print("LinearSVC  Algorithm Accuracy Percent : ",(nltk.classify.accuracy(LinearSVC_classfier,testing_set))*100)


NuSVC_classfier = SklearnClassifier(NuSVC())
NuSVC_classfier.train(traning_set)
print("NuSVC_classfier  Algorithm Accuracy Percent : ",(nltk.classify.accuracy(NuSVC_classfier,testing_set))*100)

voted_Classifier = VoteClassifier(classfier,MNB_classfier,
	B_classfier,LogisticRegression_classfier,
	SGDClassifier_classfier,LinearSVC_classfier,
	NuSVC_classfier)

print("voted_Classifier Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(voted_Classifier,testing_set))*100)
print("Classification: ",voted_Classifier.classify(testing_set[0][0]),"Confidence: ",voted_Classifier.confidence(testing_set[0][0]))
