# Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

# Load the data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')

# Clean the data
df_transform = df_raw.drop(['Cabin','PassengerId','Ticket','Name'],axis=1)
df_transform['Sex']=pd.Categorical(df_transform['Sex'])
df_transform['Embarked']=pd.Categorical(df_transform['Embarked'])
df = df_transform.copy()
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# Separate labels from features
X = df.drop(['Survived'],axis=1)
y = df['Survived']

# Encode categorical columns
X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'male' else 0)
X['Embarked'] = X['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})

# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale the data
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train[['Age','Fare']])

# Build the model
optimized_RF = RandomForestClassifier(bootstrap=False, max_depth=10, max_features='auto', min_samples_leaf=2, n_estimators=200)
optimized_RF.fit(X_train,y_train)

# Save the model
filename = '../models/Random_Forest_model.sav'
pickle.dump(optimized_RF, open(filename, 'wb'))

# Make predictions
y_pred_RF_optimized = optimized_RF.predict(X_test)
print(classification_report(y_test,y_pred_RF_optimized))