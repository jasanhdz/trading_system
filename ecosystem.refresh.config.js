module.exports = {
    apps: [
        {
            name: "04-Aegis-Turbo-Refresh-Scheduler",
            script: "aegis_alpha/tools/run_turbo_refresh_scheduler.py",
            interpreter: "/home/jasan/.venv_rocm62/bin/python",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            max_memory_restart: "512M",
            restart_delay: 30000,
            max_restarts: 10,
            kill_timeout: 60000,
            watch: false,
            env: {
                "PYTHONPATH": "/home/jasan/Develop/trading_system",
                "AEGIS_CONFIG": "aegis_alpha/configs/production.yaml",
                "AEGIS_TURBO_REFRESH_INTERVAL_SECONDS": "900",
                "AEGIS_TURBO_REFRESH_MIN_SLEEP_SECONDS": "30",
                "AEGIS_TURBO_REFRESH_MIN_AVAILABLE_MEM_GB": "8",
                "AEGIS_TURBO_REFRESH_SLEEP_BETWEEN_GROUPS_SECONDS": "5",
                "AEGIS_TURBO_REFRESH_SLEEP_BETWEEN_SYMBOLS_SECONDS": "2",
                "AEGIS_TURBO_REFRESH_GROUP_TIMEOUT_SECONDS": "600"
            },
            args: [
                "--interval-seconds", "900",
                "--min-available-mem-gb", "8",
                "--group", "ETHUSDT,BTCUSDT,SOLUSDT",
                "--group", "BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT",
                "--group", "AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT"
            ]
        }
    ]
};
