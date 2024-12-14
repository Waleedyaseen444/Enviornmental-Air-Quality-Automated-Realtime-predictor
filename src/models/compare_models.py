import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import logging
import json
import time
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelWrapper:
    """Wrapper class to standardize prediction interface"""
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
    
    def predict(self, X):
        if self.model_type == 'sklearn':
            return self.model.predict(X)
        elif self.model_type == 'lstm':
            # Convert to tensor and reshape for LSTM
            X_tensor = torch.FloatTensor(X.values)
            with torch.no_grad():
                self.model.eval()
                predictions = self.model(X_tensor)
                return predictions.numpy()
        return None

class ModelComparator:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metrics = {}
    
    def generate_test_data(self, n_samples=1000):
        """Generate test data with realistic AQI patterns"""
        np.random.seed(42)
        
        # Generate base environmental features
        data = {
            'temperature': np.random.normal(22, 5, n_samples),
            'humidity': np.random.normal(60, 10, n_samples),
            'wind_speed': np.random.normal(10, 3, n_samples),
            'pressure': np.random.normal(1013, 5, n_samples),
            'clouds': np.random.randint(0, 100, n_samples),
            'co': np.random.normal(0.5, 0.1, n_samples),
            'no2': np.random.normal(20, 5, n_samples),
            'o3': np.random.normal(30, 8, n_samples),
            'so2': np.random.normal(10, 3, n_samples),
            'pm2_5': np.random.normal(15, 5, n_samples),
            'pm10': np.random.normal(25, 8, n_samples)
        }
        
        # Add time-based features
        hours = np.random.randint(0, 24, n_samples)
        data['hour'] = hours
        data['day_of_week'] = np.random.randint(0, 7, n_samples)
        data['month'] = np.random.randint(1, 13, n_samples)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Add rolling means
        data['temperature_rolling_mean_6h'] = data['temperature'] + np.random.normal(0, 1, n_samples)
        data['humidity_rolling_mean_6h'] = data['humidity'] + np.random.normal(0, 2, n_samples)
        
        # Calculate base AQI from PM2.5 and PM10
        base_aqi = data['pm2_5'] * 0.5 + data['pm10'] * 0.5
        
        # Add AQI lags with realistic temporal correlation
        data['aqi_lag_1h'] = base_aqi + np.random.normal(0, 2, n_samples)
        data['aqi_lag_3h'] = data['aqi_lag_1h'] + np.random.normal(0, 1, n_samples)
        data['aqi_lag_6h'] = data['aqi_lag_3h'] + np.random.normal(0, 1, n_samples)
        
        # Calculate actual AQI with some additional factors
        pollution_factor = (data['co'] / 0.5 + data['no2'] / 20 + data['o3'] / 30 + data['so2'] / 10) / 4
        weather_factor = (data['temperature'] / 22 + data['humidity'] / 60) / 2
        
        data['actual_aqi'] = (base_aqi * 2 * pollution_factor * weather_factor + 
                             np.random.normal(0, 5, n_samples))
        
        # Ensure AQI values are positive and within realistic range
        data['actual_aqi'] = np.clip(data['actual_aqi'], 0, 500)
        
        logger.info(f"Generated {n_samples} test samples with realistic patterns")
        return pd.DataFrame(data)
    
    def load_models(self):
     """Load all available models"""
    try:
        # Print all files in the directory for debugging
        logger.info(f"Searching for models in: {self.models_dir}")
        all_files = list(self.models_dir.glob("*"))
        logger.info(f"All files found: {[f.name for f in all_files]}")
        
        # Load joblib models
        model_files = list(self.models_dir.glob("*.joblib"))
        logger.info(f"Model files found: {[f.name for f in model_files]}")
        
        for model_file in model_files:
            model_name = model_file.stem
            logger.info(f"Loading model: {model_name}")
            
            try:
                model = joblib.load(str(model_file))
                self.models[model_name] = ModelWrapper(model, 
                    'lstm' if 'lstm' in model_name.lower() else 'sklearn')
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
        
        if not self.models:
            logger.warning("No models were loaded!")
            
            
        logger.info(f"Successfully loaded {len(self.models)} models:")
        for name in self.models.keys():
            logger.info(f"- {name}")
        
        
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        
    
    def evaluate_model(self, model, X, y_true, model_name):
        """Evaluate a single model and log metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=model_name):
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # Additional metrics
                explained_variance = np.var(y_pred) / np.var(y_true)
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                # Calculate prediction speed
                start_time = time.time()
                model.predict(X[:100])  # Test with 100 samples
                prediction_time = (time.time() - start_time) / 100
                
                metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "explained_variance": explained_variance,
                    "mape": mape,
                    "avg_prediction_time_ms": prediction_time * 1000
                }
                
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_type", model.model_type)
                
                # Store metrics for comparison
                self.metrics[model_name] = metrics
                
                logger.info(f"\nMetrics for {model_name}:")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value:.4f}")
                
                return True
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            return False
    
    def _select_best_model(self):
        """Select the best model based on multiple metrics"""
        if not self.metrics:
            return None
            
        # Define weights for different metrics
        weights = {
            'rmse': 0.3,
            'mae': 0.2,
            'r2': 0.2,
            'mape': 0.2,
            'avg_prediction_time_ms': 0.1
        }
        
        scores = {}
        for model_name, metrics in self.metrics.items():
            # Normalize each metric (lower is better)
            normalized_metrics = {}
            for metric in weights.keys():
                if metric == 'r2':  # For R2, higher is better
                    normalized_metrics[metric] = 1 - (metrics[metric] / max(m[metric] for m in self.metrics.values()))
                else:
                    normalized_metrics[metric] = metrics[metric] / max(m[metric] for m in self.metrics.values())
            
            # Calculate weighted score
            scores[model_name] = sum(normalized_metrics[m] * w for m, w in weights.items())
        
        best_model = min(scores.items(), key=lambda x: x[1])[0]
        
        return {
            "model_name": best_model,
            "metrics": self.metrics[best_model],
            "selection_criteria": f"Composite score based on weighted metrics: {weights}"
        }

def main():
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load models
    if comparator.load_models():
        # Generate test data
        test_data = comparator.generate_test_data(n_samples=1000)
        logger.info(f"Generated test data with {len(test_data)} samples")
        
        # Prepare features and target
        X = test_data.drop('actual_aqi', axis=1)
        y = test_data['actual_aqi']
        
        # Set up MLflow
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("model_comparison")
        
        # Evaluate each model
        for model_name, model in comparator.models.items():
            comparator.evaluate_model(model, X, y, model_name)
        
        # Get best model
        best_model_info = comparator._select_best_model()
        
        if best_model_info:
            logger.info("\nBest Model Selected:")
            logger.info(f"Model Name: {best_model_info['model_name']}")
            logger.info("\nMetrics:")
            for metric, value in best_model_info['metrics'].items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Save comparison report
            report = {
                "comparison_timestamp": pd.Timestamp.now().isoformat(),
                "number_of_models_compared": len(comparator.models),
                "test_samples": len(test_data),
                "metrics_by_model": comparator.metrics,
                "best_model": best_model_info
            }
            
            report_path = comparator.models_dir / "model_comparison_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"\nDetailed comparison report saved to: {report_path}")
            logger.info("\nView detailed comparisons in MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()