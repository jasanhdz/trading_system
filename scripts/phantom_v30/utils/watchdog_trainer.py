#!/usr/bin/env python3
import os
import time
import subprocess
import json
import urllib.request
from datetime import datetime
from dotenv import load_dotenv

# Load .env explicitly
load_dotenv("/home/jasan/Develop/trading_system/binance-futures-bot-ts/.env")

# Configuration
LOG_FILE = "/home/jasan/Develop/trading_system/logs/training.log"
TIMEOUT_SECONDS = 21600  # 6 Hours (Accommodates 4h iterations of 64D model)
CHECK_INTERVAL = 900    # 15 minutes check
PM2_PROCESS_NAME = "03-V30-Trainer"

# Telegram Config (Loaded from Environment)
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_alert(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            print(f"✅ Alert sent: {response.getcode()}")
    except Exception as e:
        print(f"❌ Failed to send Telegram: {e}")

def get_file_age(filepath):
    if not os.path.exists(filepath):
        return None
    return time.time() - os.path.getmtime(filepath)

def restart_trainer():
    print(f"🔄 Restarting {PM2_PROCESS_NAME}...")
    try:
        subprocess.run(["pm2", "restart", PM2_PROCESS_NAME], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Restart failed: {e}")
        return False

def main():
    print("🛡️ Phantom Watchdog Started")
    print(f"   Monitoring: {LOG_FILE}")
    print(f"   Timeout: {TIMEOUT_SECONDS}s")
    
    while True:
        try:
            age = get_file_age(LOG_FILE)
            
            if age is None:
                print(f"⚠️ Log file missing: {LOG_FILE}")
            elif age > TIMEOUT_SECONDS:
                last_active = datetime.fromtimestamp(time.time() - age).strftime('%H:%M:%S')
                print(f"🚨 DEADLOCK! Last update: {last_active} ({int(age)}s ago)")
                
                # 1. Alert (Pre-Restart)
                send_telegram_alert(
                    f"🚨 **ALERTA DE DEADLOCK** 🚨\n"
                    f"El entrenador `03-V30-Trainer` se congeló.\n"
                    f"⏳ Inactivo por: {int(age/60)} min.\n"
                    f"🔄 **Reiniciando automáticamente...**"
                )
                
                # 2. Restart
                if restart_trainer():
                    send_telegram_alert("✅ **Reinicio Exitoso.** Dando 10 min para arrancar...")
                    time.sleep(600) # Give it 10 mins to initialize/write logs
                else:
                    send_telegram_alert("❌ **FALLO AL REINICIAR.** Revisa el servidor manualmente.")
            else:
                # print(f"✅ OK. Age: {int(age)}s")
                pass
                
        except Exception as e:
            print(f"❌ Watchdog Error: {e}")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
