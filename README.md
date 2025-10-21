# TVMaze Web Scraping & Analysis

从 TVMaze API 抓取 ≥200 条影视节目数据并生成可视化分析图表。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
```bash
python start.py
```

一键完成：
- 抓取 200 条数据 → `data/YangHong2255396.csv`
- 生成 7 幅图表 → `data/figures/`
- 记录运行日志 → `logs/scrape.log`

## 项目结构

```
tvmaze_crwal-main/
├── data/
│   ├── YangHong2255396.csv      # 抓取数据
│   └── figures/                 # 可视化图表
├── logs/
│   └── scrape.log              # 运行日志
├── src/crwal/
│   ├── scraper.py              # 数据抓取
│   ├── util.py                 # 工具函数
│   └── Visualization.py        # 数据可视化
├── start.py                    # 主程序入口
└── requirements.txt            # 依赖包
```

## 数据说明

### CSV 输出列

| 列名 | 说明 | 示例 |
|------|------|------|
| Title | 节目名称 | "Breaking Bad" |
| First air date | 首播日期（从episodes API获取最早日期） | "2008/01/20" |
| End date | 完结日期 | "2013/09/29" |
| Rating | 评分 (0-10) | 9.3 |
| Genres | 类型 (JSON) | ["Drama","Crime"] |
| Status | 播出状态 | "Ended" |
| Network | 电视网络/网络频道 | "AMC" |
| Summary | 剧情简介 | "A chemistry teacher..." |
| **Web Channel** | 网络流媒体频道（新增） | "Netflix" |
| **Language** | 节目语言（新增） | "English" |
| **Runtime** | 单集时长（分钟）（新增） | 47 |
| **Premiered Year** | 首播年份（新增） | "2008" |

### 可视化图表（7 幅）

1. `top_20_rated.png` - Top 20 评分最高节目
2. `rating_hist.png` - 评分分布直方图
3. `status_mean_rating.png` - 不同状态的平均评分
4. `genre_mean_top10.png` - Top 10 类型平均评分
5. `year_counts.png` - 节目首播年度趋势
6. `network_top15.png` - Top 15 电视网络
7. `genre_boxplot_top6.png` - Top 6 类型评分分布

## 功能特性

- ✅ 自动分页抓取数据
- ✅ 智能重试与速率限制
- ✅ 数据验证与清洗
- ✅ 实时进度显示
- ✅ 详细日志记录
- ✅ 专业可视化图表
- ✅ 中文字体自动适配

## 自定义配置

编辑 `start.py` 修改参数：

```python
COUNT = 200                                          # 抓取数量
OUT_PATH = os.path.join("data", "YangHong2255396.csv")  # 输出路径
LOG_PATH = os.path.join("logs", "scrape.log")       # 日志路径
FIG_DIR = os.path.join("data", "figures")           # 图表目录
```

修改 `src/crwal/util.py` 中的 `Config` 类调整 API 参数：

```python
@dataclass(frozen=True)
class Config:
    SLEEP_AFTER_REQ: float = 0.6    # 请求间隔（秒）
    MAX_RETRY: int = 3              # 最大重试次数
    TIMEOUT: int = 15               # 请求超时（秒）
```

## 技术栈

- **requests** - HTTP 请求
- **pandas** - 数据处理
- **matplotlib** - 数据可视化
- **tqdm** - 进度显示
- **numpy** - 数值计算

## 作业要求

| 要求 | 实现 |
|------|------|
| 抓取 ≥200 条数据 | ✅ 200 |
| 包含 8 个必需列 | ✅ 完整（额外增加4列） |
| 生成 ≥5 幅图表 | ✅ 7 幅 |
| 使用Episodes API获取首播日期 | ✅ 支持 |
| Network vs Web Channel | ✅ 独立列处理 |
| Jupyter Notebook分析 | ✅ 包含Markdown文本分析 |
| 代码规范 | ✅ 类型提示 + 工程化结构 |
| 错误处理 | ✅ 重试机制 + 异常处理 |

---

**作者**: yang hong (2255396)  
**日期**: 2025-10-21
