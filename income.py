import pandas as pd
import logitboost
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

# Load dataset
df = pd.read_csv("income.csv")
df = df.drop(df.columns[[0, 2, 4, 10, 11, 12, 13]], axis=1)

# Encode variables to integers
label_encoder = LabelEncoder()
df['class'] = df['class'].map({'<=50K': 0, '>50K': 1}).astype(int)
df['marital-status'] = df['marital-status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3,
                                                 'Divorced': 4, 'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
df.workclass = label_encoder.fit_transform(df.workclass)
df.race = label_encoder.fit_transform(df.race)
df.sex = label_encoder.fit_transform(df.sex)
df.occupation = label_encoder.fit_transform(df.occupation)
df.education = label_encoder.fit_transform(df.education)
df.relationship = label_encoder.fit_transform(df.relationship)

X = df

# Use isolation forest to get rid of possible outliers
isolation_forest = IsolationForest(contamination='auto')
isolation_forest.fit(X)
scores = isolation_forest.decision_function(X)
X['score'] = scores
filtered_df = X[X['score'] > 0]
X = filtered_df.drop('score', axis=1)

y = X.iloc[:, 7]
X = X.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=29)
lboost = logitboost.LogitBoost(random_state=29)
lboost.fit(X_train, y_train)
y_pred_train = lboost.predict(X_train)
y_pred_test = lboost.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print('Training accuracy: %.3f' % accuracy_train)
print('Test accuracy:     %.3f' % accuracy_test)
