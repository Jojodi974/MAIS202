import data_loader
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Open the pre-processed data using built-in function
file_path = "../data/S18/pre_prep/pre_prep_battlesStaging_12272020_WL_tagged.csv"
data_loader(file_path)
df_prep = joblib.load(f'data/joblib/{file_path}')

#Implement logistic regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

# Define features (X) and target (y)
X = df_prep.drop(columns=['winner.value'])
y = df_prep['winner.value']

#Separate categorical values of card.id from numerical values of average.elixir and total.level
card_columns = [
    'player1.card1.id', 'player1.card2.id', 'player1.card3.id', 'player1.card4.id',
    'player1.card5.id', 'player1.card6.id', 'player1.card7.id', 'player1.card8.id',
    'player2.card1.id', 'player2.card2.id', 'player2.card3.id', 'player2.card4.id',
    'player2.card5.id', 'player2.card6.id', 'player2.card7.id', 'player2.card8.id'
    ]
numerical_columns = ['player1.totalcard.level', 'player1.elixir.average', 'player2.totalcard.level', 'player2.elixir.average']

# Preprocessing pipeline
# Step 1: OneHotEncode card columns
# Step 2: Normalize continuous columns
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(), card_columns),  # One-hot encode card columns
        ('continuous', StandardScaler(), numerical_columns)  # Normalize continuous columns
    ])

# Split the data into training and testing sets (e.g., 80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing and logistic regression in a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))