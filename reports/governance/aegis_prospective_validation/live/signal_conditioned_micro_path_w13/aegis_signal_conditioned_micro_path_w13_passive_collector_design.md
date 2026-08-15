# W13 Passive Collector Design

This design is intentionally **not deployed**.

For each externally generated Aegis signal it records the immutable signal ID,
symbol, side, exchange timestamp and local receive timestamp, plus BOOK, QUOTE
and TRADE events from 30 seconds before through 180 seconds after the signal.
Each event retains both timestamps and its provider identity. Duplicate event
identities are ignored; synthetic microdata is prohibited.

`PassiveMicroPathCollector` is a caller-fed event sink. It contains no network
client, credentials, exchange client, PM2 definition, trading decision,
execution callback or order method. Any future deployment requires a separate
prospective-observation authorization and a durability review for storage,
restart recovery, gap handling and retention.

Dataset promotion must wait until independent TRAIN and VALIDATION minima are
met. Capturing many events around a few signals does not increase the number of
independent signal episodes.
