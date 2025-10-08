import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load Dataset
df = pd.read_csv("onlinefraud.csv")
print("✅ Dataset loaded!")

# Step 2: Add transactionID if not present
if 'transactionID' not in df.columns:
    df['transactionID'] = range(1, len(df) + 1)

# Step 3: Backup original for lookup
original_df = df.copy()

# Step 4: Prepare Features and Labels
X = df.drop(['isFraud', 'transactionID', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
y = df['isFraud']  # 🎯 Target variable for fraud detection

# Step 5: Encode categorical feature 'type'
X = pd.get_dummies(X, drop_first=True)
feature_columns = X.columns  # Save for reindexing during prediction

# Step 6: Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 8: Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 9: User Input to Check for Fraud by Transaction ID
while True:
    user_input = input("\nEnter Transaction ID to check (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break

    if not user_input.isdigit():
        print("❌ Please enter a valid numeric Transaction ID.")
        continue

    tx_id = int(user_input)
    if tx_id not in original_df['transactionID'].values:
        print("❌ Transaction ID not found.")
        continue

    # Get row for transaction ID
    row = original_df[original_df['transactionID'] == tx_id].iloc[0]

    # Extract and process input features
    features = row.drop(['isFraud', 'transactionID', 'nameOrig', 'nameDest'], errors='ignore')
    features = pd.get_dummies(features).reindex(columns=feature_columns, fill_value=0)
    features_df = features.to_frame().T if isinstance(features, pd.Series) else features
    scaled = scaler.transform(features_df)

    # Predict fraud using model
    prediction = model.predict(scaled)[0]
    actual = row['isFraud']

    # Display Results
    print("\n🔎 FRAUD DETECTION RESULT")
    print(f"🧾 Transaction ID: {tx_id}")
    print(f"📊 Model Prediction : {'🚨 FRAUDULENT (1)' if prediction == 1 else '✅ LEGITIMATE (0)'}")
    print(f"📌 Actual Label     : {actual}")
