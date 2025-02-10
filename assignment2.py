import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load datasets
train_path = 'df_train.csv'
test_path = 'df_test.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Feature selection
x_train = df_train.drop(columns=['price'])
y_train = df_train['price']
x_test = df_test.drop(columns=['price'])
y_test = df_test['price']

# Handle the date column as a numerical feature
x_train['date'] = pd.to_datetime(x_train['date'])
x_test['date'] = pd.to_datetime(x_test['date'])

# Convert 'date' to timestamp (nanoseconds since the Unix epoch)
# Convert to seconds since 1970-01-01
x_train['date_timestamp'] = x_train['date'].astype('int64') / 10**9  
x_test['date_timestamp'] = x_test['date'].astype('int64') / 10**9  

# Drop the original 'date' column
x_train = x_train.drop(columns=['date'])
x_test = x_test.drop(columns=['date'])

# Normalize input features (x_train and x_test)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Normalize the target variable 
target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1))  
y_test = target_scaler.transform(y_test.values.reshape(-1, 1))  

# Build ANN model
model = Sequential([
    Dense(16, activation='relu'),   
    Dense(16, activation='relu'), 
    Dense(1, activation='linear')  # Linear activation for regression (predict continuous values)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00005), 
              loss='mean_squared_error', 
              metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model
val_loss, val_mae = model.evaluate(x_test, y_test)
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

# Predict on the test data (model outputs normalized values)
predictions_scaled = model.predict(x_test)

# Denormalize the predictions
predictions = target_scaler.inverse_transform(predictions_scaled)

# Add predictions to the test dataframe
df_test['predicted_price'] = predictions.flatten()  # Flatten to match the original shape

# Save to CSV
df_test.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")

# Plot mae
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Training MAE', marker='o', color='b', linestyle='-', markersize=5)
plt.plot(history.history['val_mae'], label='Validation MAE', marker='x', color='r', linestyle='--', markersize=5)
plt.title('Mean Absolute Error (MAE) over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o', color='g', linestyle='-', markersize=5)
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x', color='orange', linestyle='--', markersize=5)
plt.title('Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
