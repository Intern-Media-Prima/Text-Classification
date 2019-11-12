import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load data
from os import listdir
for file in filter(lambda x: x.split('.')[-1] == 'pickle', listdir('Pickles')):
	var = file.split('.')[0]
	exec('{} = pickle.load(open("Pickles/{}", "rb"))'.format(var, file))
	print('Got {}'.format(var))
df.head()
# Let's check the dimension of our feature vectors
print(features_train.shape)
print(features_test.shape)
# n_estimators
n_estimators = [200, 800]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [10, 40]
max_depth.append(None)

# min_samples_split
min_samples_split = [10, 30, 50]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# learning rate
learning_rate = [.1, .5]

# subsample
subsample = [.5, 1.]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
			   'max_features': max_features,
			   'max_depth': max_depth,
			   'min_samples_split': min_samples_split,
			   'min_samples_leaf': min_samples_leaf,
			   'learning_rate': learning_rate,
			   'subsample': subsample}
			   
pprint(random_grid)
# First create the base model to tune
gbc = GradientBoostingClassifier(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(
	estimator=gbc,
	param_distributions=random_grid,
	n_iter=50,
	scoring='accuracy',
	cv=3, 
	verbose=1, 
	random_state=8,
	n_jobs=-1                         # Multithread job
)

# Fit the random search model
random_search.fit(features_train, labels_train)
print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)
# Create the parameter grid based on the results of random search 
max_depth = [5, 10, 15]
max_features = ['sqrt']
min_samples_leaf = [2]
min_samples_split = [50, 100]
n_estimators = [800]
learning_rate = [.1, .5]
subsample = [1.]

param_grid = {
	'max_depth': max_depth,
	'max_features': max_features,
	'min_samples_leaf': min_samples_leaf,
	'min_samples_split': min_samples_split,
	'n_estimators': n_estimators,
	'learning_rate': learning_rate,
	'subsample': subsample
	
}

# Create a base model
gbc = GradientBoostingClassifier(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(
	estimator=gbc, 
	param_grid=param_grid,
	scoring='accuracy',
	cv=cv_sets,
	verbose=1,
	n_jobs=-1
)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)
print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)
best_gbc = grid_search.best_estimator_
best_gbc
# Fit our model with training data
best_gbc.fit(features_train, labels_train)
# Predict test feature
gbc_pred = best_gbc.predict(features_test)
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_gbc.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, gbc_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test,gbc_pred))
aux_df = df[['Category', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
conf_matrix = confusion_matrix(labels_test, gbc_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
			annot=True,
			xticklabels=aux_df['Category'].values, 
			yticklabels=aux_df['Category'].values,
			cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()
# Let's see if the hyperparameter tuning process has returned a better model
base_model = GradientBoostingClassifier(random_state = 8)
base_model.fit(features_train, labels_train)
accuracy_score(labels_test, base_model.predict(features_test))
best_gbc.fit(features_train, labels_train)
accuracy_score(labels_test, best_gbc.predict(features_test))
d = {
	 'Model': 'Gradient Boosting',
	 'Training Set Accuracy': accuracy_score(labels_train, best_gbc.predict(features_train)),
	 'Test Set Accuracy': accuracy_score(labels_test, gbc_pred)
}

df_models_gbc = pd.DataFrame(d, index=[0])
df_models_gbc
# Save model and dataset
from os import mkdir
from os.path import exists
if not exists('Models'): mkdir('Models')

with open('Models/best_gbc.pickle', 'wb') as output:
	pickle.dump(best_gbc, output)
	
with open('Models/df_models_gbc.pickle', 'wb') as output:
	pickle.dump(df_models_gbc, output)
features_test

