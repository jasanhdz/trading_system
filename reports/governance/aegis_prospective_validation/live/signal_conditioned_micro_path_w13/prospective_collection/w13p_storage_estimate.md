# W13-P Storage Estimate

Observed W9.1 Tardis reference: 90 compressed files, 12,421,770,762 bytes, representing
five days x six symbols x three stream families (30 symbol-days). This is approximately:

- 414 MB per continuously captured symbol-day;
- 1.01 MB per 210-second signal window at the all-stream average;
- 1.52 GB for 1,500 non-overlapping average windows.

This linear estimate understates bursts, Parquet schema overhead, active windows across
symbols and operational margin. Capacity planning therefore reserves 5-20 GB for the
first 1,500 signals and imposes hard guards:

- stop W13-P below 100 GB free disk;
- stop W13-P at 100 GB collection size;
- never stop or backpressure trading.

Current filesystem at activation audit: 914 GB total, 242 GB used, 634 GB available.
The 90-second physical ring is memory-only, bounded and covers all active Aegis symbols.
It includes margin for journal publication delay while only T0-30s is persisted; starting
a stream after a signal cannot recover that required pre-context.
