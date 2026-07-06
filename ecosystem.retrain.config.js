const commonEnv = {
    "PYTHONPATH": "/home/jasan/Develop/trading_system",
    "AEGIS_CONFIG": "aegis_alpha/configs/production.yaml",
    "AEGIS_TURBO_MAX_SYMBOLS_PER_RUN": "1",
    "AEGIS_TURBO_MIN_AVAILABLE_MEM_GB": "12"
};

const symbols = [
    ["ETHUSDT", "2", "2,14"],
    ["BTCUSDT", "7", "2,14"],
    ["SOLUSDT", "12", "2,14"],
    ["BNBUSDT", "17", "2,14"],
    ["XRPUSDT", "22", "2,14"],
    ["DOGEUSDT", "27", "2,14"],
    ["ADAUSDT", "32", "2,14"],
    ["AVAXUSDT", "37", "2,14"],
    ["LINKUSDT", "42", "2,14"],
    ["SUIUSDT", "47", "2,14"],
    ["LTCUSDT", "52", "2,14"]
];

module.exports = {
    apps: symbols.map(([symbol, minute, hours], index) => ({
        name: `07-Aegis-Turbo-Retrain-${String(index + 1).padStart(2, "0")}-${symbol}`,
        script: "aegis_alpha/tools/run_turbo_scheduled_retrain.py",
        interpreter: "/home/jasan/.venv_rocm62/bin/python",
        cwd: "/home/jasan/Develop/trading_system",
        instances: 1,
        autostart: false,
        autorestart: false,
        cron_restart: `${minute} ${hours} * * *`,
        max_memory_restart: "6144M",
        kill_timeout: 60000,
        watch: false,
        env: commonEnv,
        args: [
            "--symbols", symbol,
            "--mode", "safe",
            "--promote-if-valid",
            "--max-symbols-per-run", "1",
            "--min-available-mem-gb", "12"
        ]
    }))
};
