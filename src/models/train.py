import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.data_path = Path("data/processed/processed_data.csv")
        self.models_path = Path("models")
        self.plots_path = Path("plots")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("pollution_prediction")
        
        # Define feature columns
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'pressure', 'clouds',
            'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
            'hour', 'day_of_week', 'month', 'is_weekend', 
            'temperature_rolling_mean_6h', 'humidity_rolling_mean_6h',
            'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h'
        ]
        self.target_column = 'aqi'

    def prepare_data(self, window_size=48):  # Increased window size to 48 hours
        """Load and prepare data for training"""
        logger.info("Loading processed data...")
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp 
        df = df.sort_values('timestamp')
        
        if len(df) < window_size + 1:
            raise ValueError(f"Insufficient data. Required window size is {window_size}, but only {len(df)} samples are available.")
        
        # Create windowed dataset
        X, y = [], []
        for i in range(window_size, len(df)):
            X.append(df[self.feature_columns].iloc[i-window_size:i].values)
            y.append(df[self.target_column].iloc[i])
        X, y = np.array(X), np.array(y)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape):
        """Build enhanced LSTM model with deeper architecture"""
        model = Sequential([
            # First LSTM layer with batch normalization
            LSTM(256, input_shape=input_shape, return_sequences=True, 
                 activation='tanh', recurrent_activation='sigmoid'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(128, return_sequences=True, 
                 activation='tanh', recurrent_activation='sigmoid'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(64, return_sequences=False,
                 activation='tanh', recurrent_activation='sigmoid'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers for final processing
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            
            Dense(1, activation='linear')
        ])
        
        # Use a slightly lower learning rate for better stability
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def plot_training_history(self, history, timestamp):
        """Plot and save training history"""
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = self.plots_path / f"training_history_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

    def train_lstm_model(self, window_size=48, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test = self.prepare_data(window_size)
        
        logger.info("Training LSTM model...")
        
        model = self.build_lstm_model((window_size, X_train.shape[2]))
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot and save training history
        plot_path = self.plot_training_history(history, timestamp)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics and model with MLflow
        with mlflow.start_run(run_name=f"LSTM_{timestamp}"):
            mlflow.log_param("window_size", window_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("initial_epochs", epochs)
            mlflow.log_param("actual_epochs", len(history.history['loss']))
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log the training history plot
            mlflow.log_artifact(str(plot_path))
            
            # Log the model
            mlflow.keras.log_model(model, "lstm_model")
            
            logger.info(f"LSTM metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        # Save the trained model
        model_path = self.models_path / f"lstm_model_{timestamp}.h5"
        model.save(model_path)
        logger.info(f"LSTM model saved to {model_path}")
        logger.info(f"Training history plot saved to {plot_path}")

def main():
    trainer = ModelTrainer()
    trainer.train_lstm_model()

if __name__ == "__main__":
    main()