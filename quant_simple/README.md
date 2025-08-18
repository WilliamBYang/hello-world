## 简介
一个简单的移动平均线金叉/死叉回测示例（Python）。可从雅虎财经抓取数据，或读取本地 CSV（列包含 `Date, Close`）。支持交易费、保存结果与绘图。

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使用
- 抓取数据并回测：
```bash
python main.py --symbol AAPL --start 2022-01-01 --end 2024-12-31 --short 20 --long 50 --plot
```

- 使用本地 CSV：
```bash
python main.py --csv path/to/data.csv --short 10 --long 30
```
CSV 需包含列：`Date, Close`。

## 输出
- 终端打印：收益、波动率、夏普、回撤等指标
- 保存 `backtest_<symbol>.csv`
- 若加 `--plot`，保存 `equity_<symbol>.png`

## 免责声明
仅用于学习演示，不构成投资建议。