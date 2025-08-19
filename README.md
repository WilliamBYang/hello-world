# Barra Extended (with Skewness, Kurtosis, VaR, ES)

## Setup
```bash
pip install -r requirements.txt
```

## Run demo
```bash
python barra_extended.py
```

## Data schema
Input panel DataFrame should contain columns:
- `date`: timestamp or string
- `ticker`: string
- `close`: float
- `returns`: float (simple return per day)
- `mcap`: float
- `book_equity`: float
- `roe`: float
- `turnover`: float
- `industry`: string

Replace the synthetic generator with your own data to run in production.
