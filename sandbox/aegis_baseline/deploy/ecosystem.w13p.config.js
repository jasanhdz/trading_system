module.exports = {
  apps: [
    {
      name: 'w13p-passive-microstructure-collector',
      script: '/home/jasan/.venv_rocm62/bin/python',
      args: '-m aegis.research.prospective_microstructure_w13p collect --config config/experiments/aegis_w13p_prospective_collection.yaml',
      interpreter: 'none',
      cwd: '/home/jasan/Develop/trading_system/sandbox/aegis',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 20,
      max_memory_restart: '3072M',
      watch: false,
      kill_timeout: 15000,
      env: {
        PYTHONPATH: '/home/jasan/Develop/trading_system/sandbox/aegis_baseline/src',
        PYTHONUNBUFFERED: '1',
      },
      out_file: '/home/jasan/Develop/trading_system/data/w13p_prospective_collection/runtime/stdout.log',
      error_file: '/home/jasan/Develop/trading_system/data/w13p_prospective_collection/runtime/stderr.log',
      merge_logs: true,
    },
  ],
};
