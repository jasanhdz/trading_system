module.exports = {
    apps: [
        {
            name: "01-Trading-Bot",
            script: "dist/main.js",
            cwd: "/home/jasan/Develop/trading_system/binance-futures-bot-ts",
            instances: 1,
            exec_mode: "fork",
            autorestart: true,
            watch: false,
            node_args: "-r dotenv/config",
            env: {
                "NODE_ENV": "production",
                "DOTENV_CONFIG_PATH": ".env"
            }
        },
        {
            name: "02-Phantom-API",
            script: "scripts/phantom_v30/inference_server.py",
            interpreter: ".venv_rocm62/bin/python",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            watch: false,
            env: {
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                "LD_LIBRARY_PATH": "/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64",
                "HIP_VISIBLE_DEVICES": ""  // CPU-only inference, frees GPUs for training
            }
        },
        {
            name: "03-V30-Trainer",
            script: "scripts/phantom_v30/matrix_trainer.py",
            interpreter: ".venv_rocm62/bin/python",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            watch: false,
            env: {
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                "LD_LIBRARY_PATH": "/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64",
                "HIP_VISIBLE_DEVICES": "0,1",  // Both GPUs for dual training
                "DATABASE_URL": "sqlite:////home/jasan/Develop/trading_system/data/binance_candles.db"
            }
        },
        {
            name: "04-Exit-V2-Trainer",
            script: "scripts/phantom_v30/train_exit_agent_v2.py",
            interpreter: ".venv_rocm62/bin/python",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            watch: false,
            env: {
                // Must be CPU only to avoid blocking V30 Trainer on GPU
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "DATABASE_URL": "sqlite:////home/jasan/Develop/trading_system/data/binance_candles.db"
            }
        },
        {
            name: "99-Watchdog",
            script: "scripts/phantom_v30/watchdog_trainer.py",
            interpreter: ".venv_rocm62/bin/python",
            cwd: "/home/jasan/Develop/trading_system",
            instances: 1,
            autorestart: true,
            watch: false,
            // Load environment variables from .env file
            // Note: This requires PM2 to support 'env_file' or requires manual sourcing.
            // If this fails, we might need to use python-dotenv.
            env: {
                "PYTHONUNBUFFERED": "1"
            },
            env_file: "binance-futures-bot-ts/.env"
        }
    ]
};
