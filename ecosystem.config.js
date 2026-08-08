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
            // The complete observational committee peaks near 7.4 GiB at startup.
            // Keep it loaded so the Live API and Shadow evidence remain identical.
            max_memory_restart: "10240M",
            restart_delay: 10000,
            max_restarts: 10,
            watch: false,
            env: {
                "PYTHONPATH": "/home/jasan/Develop/trading_system/src",
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                "LD_LIBRARY_PATH": "/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64",
                "HIP_VISIBLE_DEVICES": ""  // CPU-only inference, frees GPUs for training
            }
        }
    ]
};
