from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

df = pd.read_csv("onlinefraud.csv")
if 'transactionID' not in df.columns:
    df['transactionID'] = range(1, len(df) + 1)

original_df = df.copy()

X = df.drop(['isFraud', 'transactionID', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
y = df['isFraud']
X = pd.get_dummies(X, drop_first=True)
feature_columns = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        tx_id_input = request.form.get('transaction_id')
        if not tx_id_input.isdigit():
            result = {'error': '❌ Please enter a numeric Transaction ID.'}
        else:
            tx_id = int(tx_id_input)
            if tx_id not in original_df['transactionID'].values:
                result = {'error': '❌ Transaction ID not found in dataset.'}
            else:
                row = original_df[original_df['transactionID'] == tx_id].iloc[0]
                features = row.drop(['isFraud', 'transactionID', 'nameOrig', 'nameDest'], errors='ignore')
                features_df = features.to_frame().T
                features_encoded = pd.get_dummies(features_df)
                features_encoded = features_encoded.reindex(columns=feature_columns, fill_value=0)
                scaled_input = scaler.transform(features_encoded)
                prediction = model.predict(scaled_input)[0]
                actual = row['isFraud']
                result = {
                    'tx_id': tx_id,
                    'prediction': f"{prediction} ({'🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE'})",
                    'actual': f"{actual} (from dataset)",
                    'match': prediction == actual
                }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
