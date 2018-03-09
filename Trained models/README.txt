Note:

Feature Sets needs to trained and pickled, use traning_models script to train various classifiers along with feature sets.

sample code:
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]
save_word_features = open("word_features.pickle","wb")
pickle.dump(documents,save_word_features)
save_word_features.close()

