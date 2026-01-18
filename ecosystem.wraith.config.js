module.exports = {
    apps: [
        {
            name: "99-Wraith-Shadow",
            script: "./scripts/wraith_shadow_service.py",
            interpreter: "./.venv_rocm62/bin/python3",
            watch: false,
            autorestart: true,
            restart_delay: 5000,
            env: {
                PYTHONUNBUFFERED: "1",
                NODE_ENV: "production"
            }
        }
    ]
};
