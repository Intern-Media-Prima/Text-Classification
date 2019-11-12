import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pickle.load(open('df.pickle', 'rb'))
trainX, testX, trainY, testY = train_test_split(df['Content_Parsed'], df['Category_Code'], test_size=0.15, random_state=8)

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1, 2),
                        stop_words=None,
                        lowercase=False,
                        max_df=1.,
                        min_df=1,
                        max_features=300,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(trainX).toarray()
labels_train = trainY
print(features_train.shape)

features_test = tfidf.transform(testX).toarray()
labels_test = testY
print(features_test.shape)

# Load our model
gbc = pickle.load(open('Models/best_gbc.pickle', 'rb'))

# Fit our model with training data
gbc.fit(features_train, labels_train)
print('Done fitting model with training data')

# Predict test feature
print('Predicting model with testing data')
gbc_pred = gbc.predict(features_test)

# Training accuracy
print("The training accuracy is: {}".format(accuracy_score(labels_train, gbc.predict(features_train))))

# Test accuracy
print("The test accuracy is: {}".format(accuracy_score(labels_test, gbc_pred)))

# Classification report
print("Classification report\n{}".format(classification_report(labels_test,gbc_pred)))

with open('Models/gbc.pickle', 'wb') as file:
	file.write(pickle.dumps(gbc))
	print('Model successfully saved!')