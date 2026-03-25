import requests

try:
    resp = requests.post("http://localhost:8001/ml-v2/exit_signal", json={
        "symbol": "ETHUSDT",
        "entry_price": 2250.0,
        "current_pnl": 0.05,
        "mfe": 0.10,
        "mae": -0.05,
        "duration_minutes": 60,
        "leverage": 20
    })
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print(e)
