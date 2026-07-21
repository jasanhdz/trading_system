module.exports = {
  apps: [
    {
      name: 'aegis-prospective-shadow-cohort-1',
      cwd: '/home/jasan/Develop/trading_system',
      script: 'scripts/run_aegis_shadow_cohort_1.sh',
      interpreter: '/bin/bash',
      autorestart: true,
      restart_delay: 1000,
      exp_backoff_restart_delay: 1000,
      max_restarts: 5,
      min_uptime: '30s',
      kill_timeout: 30000,
      listen_timeout: 30000,
      max_memory_restart: '1G',
      out_file: 'data/prospective_shadow/cohort_1/logs/service.stdout.log',
      error_file: 'data/prospective_shadow/cohort_1/logs/service.stderr.log',
      merge_logs: true,
      time: false,
      env: {},
    },
  ],
};
