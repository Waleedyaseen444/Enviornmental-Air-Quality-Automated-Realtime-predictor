from fastapi import FastAPI
import requests
import time

def test_metrics_emission():
    """
    Test function to verify metrics are being properly emitted
    Returns a dictionary with test results
    """
    try:
        # Test metrics endpoint
        response = requests.get('http://localhost:8000/metrics')
        if response.status_code != 200:
            return {"success": False, "error": f"Metrics endpoint returned {response.status_code}"}
        
        metrics_text = response.text
        
        # Check for specific metrics that should be present
        expected_metrics = [
            'prediction_error',
            'model_prediction_duration_seconds',
            'api_health'
        ]
        
        missing_metrics = []
        for metric in expected_metrics:
            if metric not in metrics_text:
                missing_metrics.append(metric)
        
        if missing_metrics:
            return {
                "success": False, 
                "error": f"Missing metrics: {', '.join(missing_metrics)}"
            }
            
        # Test gauge metrics specifically
        gauge_metrics = [
            'prediction_error',
            'api_health'
        ]
        
        # Make a test prediction to generate some metrics
        test_data = {
            "city": "London",
            "temperature": 20.5,
            "humidity": 65,
            "wind_speed": 5.2,
            "pressure": 1013,
            "clouds": 75,
            "co": 0.5,
            "no2": 0.04,
            "o3": 0.06,
            "so2": 0.02,
            "pm2_5": 15,
            "pm10": 25
        }
        
        pred_response = requests.post('http://localhost:8000/predict', json=test_data)
        if pred_response.status_code != 200:
            return {"success": False, "error": "Failed to generate test prediction"}
            
        # Wait briefly for metrics to update
        time.sleep(2)
        
        # Check metrics again
        response = requests.get('http://localhost:8000/metrics')
        metrics_text = response.text
        
        # Verify gauge metrics are present with values
        for metric in gauge_metrics:
            if f'{metric} ' not in metrics_text:  # Space after metric name to ensure exact match
                return {
                    "success": False,
                    "error": f"Gauge metric {metric} not found after test prediction"
                }
        
        return {"success": True, "message": "All metrics are being emitted correctly"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    results = test_metrics_emission()
    print(results)