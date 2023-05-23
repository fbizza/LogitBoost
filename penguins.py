import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logitboost
import warnings

warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df = pd.DataFrame(data)

# Encode strings features to integers
label_encoder = LabelEncoder()
df.sex = label_encoder.fit_transform(df.sex)  # TODO: check if it is a relevant feature
df.island = label_encoder.fit_transform(df.island)
df.species = label_encoder.fit_transform(df.species)

# Separate features and labels
X = df.iloc[:, 1:]
species = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, species, test_size=0.30, shuffle=True, random_state=29)

# Fit and predict
lboost = logitboost.LogitBoost(random_state=29)
lboost.fit(X_train, y_train)
y_pred_train = lboost.predict(X_train)
y_pred_test = lboost.predict(X_test)

# Evaluation
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print('Training accuracy: %.2f' % accuracy_train)
print('Test accuracy:     %.2f' % accuracy_test)
