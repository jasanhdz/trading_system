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
                "PHANTOM_LEVERAGE": "5",
                "PHANTOM_POSITION_FRACTION": "0.25",
                "PHANTOM_HARD_STOP_ROE": "0.15",
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
                "PHANTOM_LEVERAGE": "5",
                "PHANTOM_POSITION_FRACTION": "0.25",
                "PHANTOM_HARD_STOP_ROE": "0.15",
                "PHANTOM_CHALLENGER_A_SOURCE": "champion",
                "PHANTOM_CHALLENGER_B_SOURCE": "bc",
                "PHANTOM_CHAMPION_MUTATION_ENTROPY": "0.10",
                "PHANTOM_CHAMPION_MUTATION_LR": "0.00025",
                "PHANTOM_LATEST_MUTATION_ENTROPY": "0.10",
                "PHANTOM_BC_MUTATION_ENTROPY": "0.12",
                "PHANTOM_FRESH_MUTATION_ENTROPY": "0.18",
                "PHANTOM_ENTROPY_FLOOR": "0.01",
                "PHANTOM_EARLY_CLOSE_HOLD_STEPS": "3",
                "PHANTOM_EARLY_CLOSE_PENALTY": "0.08",
                "PHANTOM_RAPID_TURNOVER_HOLD_STEPS": "6",
                "PHANTOM_RAPID_TURNOVER_PENALTY": "0.04",
                "PHANTOM_REENTRY_PENALTY": "0.08",
                "PHANTOM_MIN_HOLD_STEPS": "6",
                "PHANTOM_MIN_FLAT_STEPS": "6",
                "PHANTOM_INVALID_ACTION_PENALTY": "-0.02",
                "PHANTOM_EVAL_NUM_ENVS": "64",
                "PHANTOM_EVAL_MAX_STEPS": "4032",
                "PHANTOM_SIGNALQ_SAMPLE_EVERY": "24",
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
                "PHANTOM_LEVERAGE": "5",
                "PHANTOM_POSITION_FRACTION": "0.25",
                "PHANTOM_HARD_STOP_ROE": "0.15",
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
            restart_delay: 10000,
            max_restarts: 10,
            watch: false,
            // Load environment variables from .env file
            // Note: This requires PM2 to support 'env_file' or requires manual sourcing.
            // If this fails, we might need to use python-dotenv.
            env: {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/home/jasan/Develop/trading_system",
                "WATCHDOG_CHECK_INTERVAL": "60",
                "IO_FULL_AVG10_STOP_THRESHOLD": "50",
                "IO_SOME_AVG10_STOP_THRESHOLD": "65",
                "IO_PRESSURE_CONSECUTIVE_LIMIT": "3",
                "IO_PRESSURE_STOP_ENABLED": "false",
                "IO_PRESSURE_ALERT_COOLDOWN_SECONDS": "1800"
            },
            env_file: "binance-futures-bot-ts/.env"
        }
    ]
};
