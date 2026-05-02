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
CHECK_INTERVAL = int(os.environ.get("WATCHDOG_CHECK_INTERVAL", "60"))
PM2_PROCESS_NAME = "03-V30-Trainer"
IO_FULL_AVG10_STOP_THRESHOLD = float(os.environ.get("IO_FULL_AVG10_STOP_THRESHOLD", "50"))
IO_SOME_AVG10_STOP_THRESHOLD = float(os.environ.get("IO_SOME_AVG10_STOP_THRESHOLD", "65"))
IO_PRESSURE_CONSECUTIVE_LIMIT = int(os.environ.get("IO_PRESSURE_CONSECUTIVE_LIMIT", "3"))
IO_PRESSURE_STOP_ENABLED = os.environ.get("IO_PRESSURE_STOP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
IO_PRESSURE_ALERT_COOLDOWN_SECONDS = int(os.environ.get("IO_PRESSURE_ALERT_COOLDOWN_SECONDS", "1800"))
PM2_LOG_DIR = os.environ.get("PM2_LOG_DIR", "/home/jasan/.pm2/logs")
CHECKPOINT_DIR = "/home/jasan/Develop/trading_system/models/checkpoints"

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

def stop_trainer():
    print(f"⏸️ Stopping {PM2_PROCESS_NAME} to let disk I/O recover...")
    try:
        subprocess.run(["pm2", "stop", PM2_PROCESS_NAME], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Stop failed: {e}")
        return False

def read_io_pressure():
    try:
        with open("/proc/pressure/io", "r", encoding="utf-8") as f:
            values = {}
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                category = parts[0]
                metrics = {}
                for item in parts[1:]:
                    key, value = item.split("=", 1)
                    metrics[key] = float(value)
                values[category] = metrics
            return values
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"❌ Failed to read I/O pressure: {e}")
        return {}

def directory_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total

def format_bytes(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"

def collect_io_context():
    lines = []

    try:
        log_files = []
        now = time.time()
        if os.path.isdir(PM2_LOG_DIR):
            for entry in os.scandir(PM2_LOG_DIR):
                if not entry.is_file():
                    continue
                stat = entry.stat()
                age_minutes = max(0.0, (now - stat.st_mtime) / 60.0)
                log_files.append((stat.st_size, age_minutes, entry.path))
        for size, age_minutes, path in sorted(log_files, reverse=True)[:5]:
            lines.append(f"• PM2 log: `{os.path.basename(path)}` {format_bytes(size)} (mtime {age_minutes:.0f}m)")
    except Exception as e:
        lines.append(f"• PM2 log scan failed: `{e}`")

    checkpoint_bytes = directory_size(CHECKPOINT_DIR)
    lines.append(f"• Checkpoints: {format_bytes(checkpoint_bytes)}")
    return "\n".join(lines)

def main():
    print("🛡️ Phantom Watchdog Started")
    print(f"   Monitoring: {LOG_FILE}")
    print(f"   Timeout: {TIMEOUT_SECONDS}s")
    print(f"   Check interval: {CHECK_INTERVAL}s")
    print(f"   I/O guard: full.avg10>={IO_FULL_AVG10_STOP_THRESHOLD} or some.avg10>={IO_SOME_AVG10_STOP_THRESHOLD} for {IO_PRESSURE_CONSECUTIVE_LIMIT} checks")
    print(f"   I/O stop enabled: {IO_PRESSURE_STOP_ENABLED}")
    io_pressure_hits = 0
    last_io_alert_at = 0.0
    
    while True:
        try:
            pressure = read_io_pressure()
            full_avg10 = pressure.get("full", {}).get("avg10", 0.0)
            some_avg10 = pressure.get("some", {}).get("avg10", 0.0)
            if full_avg10 >= IO_FULL_AVG10_STOP_THRESHOLD or some_avg10 >= IO_SOME_AVG10_STOP_THRESHOLD:
                io_pressure_hits += 1
            else:
                io_pressure_hits = 0

            if io_pressure_hits >= IO_PRESSURE_CONSECUTIVE_LIMIT:
                context = collect_io_context()
                msg = (
                    f"🚨 **ALERTA DE I/O** 🚨\n"
                    f"Presión de disco sostenida: full.avg10={full_avg10:.1f}, some.avg10={some_avg10:.1f}.\n"
                    f"{context}\n"
                )
                print(msg)
                now = time.time()
                if IO_PRESSURE_STOP_ENABLED:
                    send_telegram_alert(msg + f"⏸️ Deteniendo `{PM2_PROCESS_NAME}` para prevenir congelamiento del host.")
                    if stop_trainer():
                        send_telegram_alert("✅ Trainer detenido. Revisa disco/checkpoints antes de reactivarlo.")
                    io_pressure_hits = 0
                    time.sleep(600)
                    continue

                if now - last_io_alert_at >= IO_PRESSURE_ALERT_COOLDOWN_SECONDS:
                    send_telegram_alert(
                        msg
                        + f"ℹ️ No se detuvo `{PM2_PROCESS_NAME}` porque `IO_PRESSURE_STOP_ENABLED=false`."
                    )
                    last_io_alert_at = now
                io_pressure_hits = 0

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
