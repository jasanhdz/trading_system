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
            script: "aegis_alpha/inference/server.py",
            interpreter: "/home/jasan/.venv_rocm62/bin/python",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            max_memory_restart: "4096M",
            restart_delay: 10000,
            max_restarts: 10,
            watch: false,
            env: {
                "PYTHONPATH": "/home/jasan/Develop/trading_system",
                "AEGIS_CONFIG": "aegis_alpha/configs/production.yaml",
                "AEGIS_PORT": "8001",
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                "LD_LIBRARY_PATH": "/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64",
                "HIP_VISIBLE_DEVICES": ""  // CPU-only inference, frees GPUs for training
            }
        }
    ]
};
