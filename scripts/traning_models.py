#Building voting system system on feature set to increase accuracy and realiablity of the classifier
#Building our own classifier by compling varoius classifiers based on voting of various classifiers

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.tokenize import word_tokenize

#wrapper of Sklearn Around nltk classifiers
from nltk.classify.scikitlearn import SklearnClassifier

#importing various sckitlearn clsssifiers and testing them on a movie_review document
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


short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents =[]
all_words =[]

allowed_words =["J","V","R"]
for p in short_pos.split("\n"):
	documents.append((p,"pos"))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_words:
			all_words.append(w[0].lower)

for p in short_neg.split("\n"):
	documents.append((p,"pos"))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_words:
			all_words.append(w[0].lower)

save_documents = open("documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_word_features = open("word_features.pickle","wb")
pickle.dump(documents,save_word_features)
save_word_features.close()

def find_features(document): #creating dictinary of word list in document tuple with boolean as a value
	words= word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words) # if present in most common 3000 words set it as true

	return features

#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets = [(find_features(rev),category) for (rev,category)  in documents] # create a feature set with all words in document with category
random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[10000:]
testing_set = featuresets[:10000]

# Various Classifiers and saving them to pickle

classfier = nltk.NaiveBayesClassifier.train(traning_set) #using naive bayes algo to classify pos or neg movie reviews
save_classfier = open("NaiveBayes.pickle","wb") #saving the trained model using pickle
pickle.dump(classfier,save_classfier)
save_classfier.close()
print("ORiginal Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(classfier,testing_set))*100) #calculating accuracy of th model


MNB_classfier = SklearnClassifier(MultinomialNB())
MNB_classfier.train(training_set)
print("Multinomial Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(MNB_classfier,testing_set))*100) #calculating accuracy of th model
save_classfierMNB = open("MNB_classfier.pickle","wb") #saving the trained model using pickle
pickle.dump(MNB_classfier,save_classfierMNB)
save_classfierMNB.close()


B_classfier = SklearnClassifier(BernoulliNB())
B_classfier.train(training_set)
print("Bernoulli Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(B_classfier,testing_set))*100) #calculating accuracy of th model
save_classfierB = open("B_classfier.pickle","wb") #saving the trained model using pickle
pickle.dump(B_classfier,save_classfierB)
save_classfierB.close()

#LogisticRegression,SGDClassifier
#SVC,LinearSVC,NuSVC

LogisticRegression_classfier = SklearnClassifier(LogisticRegression())
LogisticRegression_classfier.train(training_set)
print("LogisticRegression Algorithm Accuracy Percent : ",(nltk.classify.accuracy(LogisticRegression_classfier,testing_set))*100)
save_LogisticRegression_classfier = open("LogisticRegression_classfier.pickle","wb") #saving the trained model using pickle
pickle.dump(LogisticRegression_classfier,save_classfierB)
save_LogisticRegression_classfier.close()

SGDClassifier_classfier = SklearnClassifier(SGDClassifier())
SGDClassifier_classfier.train(training_set)
print("SGDClassifier_classfier  Algorithm Accuracy Percent : ",(nltk.classify.accuracy(SGDClassifier_classfier,testing_set))*100)
save_SGDClassifier_classfier = open("SGDClassifier_classfier.pickle","wb") #saving the trained model using pickle
pickle.dump(SGDClassifier_classfier,save_classfierB)
save_SGDClassifier_classfier.close()


LinearSVC_classfier = SklearnClassifier(LinearSVC())
LinearSVC_classfier.train(training_set)
print("LinearSVC  Algorithm Accuracy Percent : ",(nltk.classify.accuracy(LinearSVC_classfier,testing_set))*100)
save_LinearSVC_classfier = open("LinearSVC_classfier.pickle","wb") #saving the trained model using pickle
pickle.dump(LinearSVC_classfier,save_classfierB)
save_LinearSVC_classfier.close()

#NuSVC_classfier = SklearnClassifier(NuSVC())
#NuSVC_classfier.train(traning_set)
#print("NuSVC_classfier  Algorithm Accuracy Percent : ",(nltk.classify.accuracy(NuSVC_classfier,testing_set))*100)

voted_Classifier = VoteClassifier(classfier,MNB_classfier,
	B_classfier,LogisticRegression_classfier,
	SGDClassifier_classfier,LinearSVC_classfier)
print("voted_Classifier Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(voted_Classifier,testing_set))*100)
