# Local Data

`raw_prices.csv` contains pre-downloaded daily Yahoo Finance OHLCV data for:

- SPY
- AAPL
- MSFT
- NVDA
- JPM

The notebook loads this file directly and does not call Yahoo Finance or any other external data API during execution. Columns are:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

Date range: 2014-01-02 through 2026-04-21.
