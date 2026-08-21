module.exports = {
    apps: [
        {
            name: "01-Trading-Bot",
            script: "dist/main.js",
            cwd: "/home/jasan/Develop/trading_system/binance-futures-bot-ts",
            instances: 1,
            exec_mode: "fork",
            autorestart: true,
            max_memory_restart: "1024M",
            watch: false,
            node_args: "-r dotenv/config",
            env: {
                "NODE_ENV": "production",
                "DOTENV_CONFIG_PATH": ".env"
            }
        },
        {
            name: "02-Aegis-API",
            script: "/home/jasan/.venv_rocm62/bin/python",
            args: "-m aegis.live_api --host 127.0.0.1 --port 8001",
            interpreter: "none",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            // Current Brain plus full-history E4 peaks near 12.3 GiB at startup.
            // Preserve headroom without masking unbounded growth.
            max_memory_restart: "14336M",
            restart_delay: 10000,
            max_restarts: 10,
            watch: false,
            env: {
                "PYTHONPATH": "/home/jasan/Develop/trading_system/src",
                "E4_HISTORY_SEED_ROOT": "/home/jasan/Develop/trading_system/data/independent_entry_quality_discovery_v1/candles_1m",
                "E4_DURABLE_CACHE_ROOT": "/home/jasan/Develop/trading_system/data/e4_live_candle_cache",
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                "LD_LIBRARY_PATH": "/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64",
                "HIP_VISIBLE_DEVICES": ""  // CPU-only inference, frees GPUs for training
            }
        }
    ]
};
