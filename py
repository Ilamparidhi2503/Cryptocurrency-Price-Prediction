import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Load datasets
df1 = pd.read_csv("C:/Users/Ilamparidhi/Downloads/Cryptocurrency Prices Dataset.csv")
df2 = pd.read_csv("C:/Users/Ilamparidhi/Downloads/Cryptocurrency Prices Dataset v1.csv")

# Merging datasets
common_cols = list(set(df1.columns) & set(df2.columns))
if common_cols:
    df = pd.merge(df1, df2, on=common_cols, how='outer')
else:
    df = pd.concat([df1, df2], axis=0)

# Print column names to debug
print("Available Columns:", df.columns)

# Handling missing values
df.ffill(inplace=True)

# Convert date column to datetime and set as index
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

# Identify potential price column
price_column = next((col for col in df.columns if "price" in col.lower()), None)
if not price_column:
    raise ValueError("No column containing 'price' found in dataset!")

print(f"Using '{price_column}' as the target column.")

# Convert selected price column to numeric
df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
df.dropna(subset=[price_column], inplace=True)

# Selecting the target variable
data = df[[price_column]].copy()  # Fix: Use .copy() to avoid SettingWithCopyWarning

# Normalize data
scaler = MinMaxScaler()
price_scaler = MinMaxScaler()
data[price_column] = price_scaler.fit_transform(data[[price_column]])

# Create sequences for LSTM
sequence_length = 50
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(data.values, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model (Fix: Use Input Layer)
model = Sequential([
    Input(shape=(sequence_length, X.shape[2])),  # Fix: Proper input definition
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)

# Apply inverse transformation
y_pred = price_scaler.inverse_transform(y_pred)
y_test = price_scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(14,5))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.legend()
plt.title('Cryptocurrency Price Prediction')
plt.show()
