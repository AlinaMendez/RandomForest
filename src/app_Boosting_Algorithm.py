# Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Load the data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')

# Clean the data
df = df_raw.drop(['Cabin','PassengerId','Ticket','Name'],axis=1)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df['Embarked'] = df['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})

# Separate labels from features
X = df.drop(['Survived'],axis=1)
y = df['Survived']

# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale the data
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train[['Age','Fare']])

# Make the model
model = xgb.XGBClassifier(colsample_bytree= 0.7, eta=0.05, gamma=0.0, max_depth=5, min_child_weight=1)
model.fit(X_train, y_train)

# Save the model
filename = '../models/Boosting_Algorithm_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Make predictions
y_pred_BA = model.predict(X_test)
print(classification_report(y_test,y_pred_BA))
print(accuracy_score(y_test,y_pred_BA))