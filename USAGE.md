# 使用说明

## 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行程序（一键完成数据抓取和可视化）
python start.py
```

## 输出结果

运行完成后，将生成以下文件：

### 数据文件
- `data/YangHong2255396.csv` - 包含 200 条 TV 节目数据

### 可视化图表（7 幅）
- `data/figures/top_20_rated.png` - Top 20 评分最高节目
- `data/figures/rating_hist.png` - 评分分布直方图
- `data/figures/status_mean_rating.png` - 不同状态的平均评分
- `data/figures/genre_mean_top10.png` - Top 10 类型平均评分
- `data/figures/year_counts.png` - 节目首播年度趋势
- `data/figures/network_top15.png` - Top 15 电视网络
- `data/figures/genre_boxplot_top6.png` - Top 6 类型评分分布

### 日志文件
- `logs/scrape.log` - 详细的运行日志

## 修改配置

编辑 `start.py` 可以修改以下参数：

```python
COUNT = 200        # 抓取数量（可改为 300, 500 等）
OUT_PATH = ...     # CSV 输出路径
LOG_PATH = ...     # 日志文件路径
FIG_DIR = ...      # 图表输出目录
```

## 代码结构

```
总计: 505 行代码
├── util.py (114 行) - 工具函数和配置
├── scraper.py (95 行) - 数据抓取逻辑
├── Visualization.py (276 行) - 7 个可视化函数
└── start.py (20 行) - 主程序入口
```

## 特性

- ✅ 无冗余代码，无注释，简洁高效
- ✅ 符合工程规范，使用类型提示
- ✅ 一键运行，自动完成抓取和可视化
- ✅ 自动重试和错误处理
- ✅ 进度条显示
- ✅ 详细日志记录

