module.exports = {
  apps: [
    {
      name: "06-Golden-Sniper",
      script: "./scripts/golden_opportunity_service.py",
      interpreter: "./.venv_cuda/bin/python3",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1",
        NODE_ENV: "production"
      }
    }
  ]
};
