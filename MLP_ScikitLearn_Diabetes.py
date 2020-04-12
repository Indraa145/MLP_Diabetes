# Import required libraries & necessary modules
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Getting the data
df = pd.read_csv('data/diabetes.csv')

#Setting some variables
target_column = ['Outcome']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max() #Normalizing the values for X
print(df.describe().transpose())

#Creating the Training and Test Data
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

#Building the Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Training set accuracy: %f" % mlp.score(X_train, y_train))
print("Test set accuracy: %f" % mlp.score(X_test, y_test))

#Compare Training data result & original
print("For Training Data:")
print("Confusion Matrix : ")
print(confusion_matrix(y_train, predict_train))
print("Classification Report : ")
print(classification_report(y_train, predict_train))

#Compare Test data result & original
print("For Test Data:")
print("Confusion Matrix : ")
print(confusion_matrix(y_test, predict_test))
print("Classification Report : ")
print(classification_report(y_test, predict_test))