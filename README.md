import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# Step 1: Data collection and preprocessing
data = pd.read_csv('credit_card_transactions.csv')  # Replace with your dataset path
# Perform any necessary data cleaning and exploration here

# Step 2: Feature engineering
# Extract relevant features from the data

# Step 3: Model training
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a random forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train an XGBoost classifier
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Step 4: Model evaluation
# Make predictions on the test set
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the models
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_predictions))

print("XGBoost Classifier Report:")
print(classification_report(y_test, xgb_predictions))

# Step 5: Fraud detection and monitoring
# Apply the trained models to new, unseen transactions for fraud detection
new_transactions = pd.read_csv('new_transactions.csv')  # Replace with your new transaction data
new_transactions = scaler.transform(new_transactions)

rf_predictions_new = rf_model.predict(new_transactions)
xgb_predictions_new = xgb_model.predict(new_transactions)

# Perform further actions based on the predictions
# such as flagging suspicious transactions or sending alerts

