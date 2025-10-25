# TVMaze 爬虫与数据分析

从 TVMaze 网站抓取电视节目数据并进行可视化分析。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
```bash
python start.py
```

程序会自动：
- 抓取数据并保存到 `data/日期/YangHong2255396.csv`
- 生成图表到 `data/日期/figures/`
- 记录日志到 `logs/scrape.log`

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
| First air date | 首播日期 | "2008/01/20" |
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

### 生成的图表

1. `top_20_rated.png` - 评分最高的20个节目
2. `rating_hist.png` - 评分分布直方图
3. `status_mean_rating.png` - 不同状态的平均评分
4. `genre_mean_top10.png` - 各类型平均评分
5. `year_counts.png` - 节目首播年份趋势
6. `network_top15.png` - 主要电视网络
7. `genre_boxplot_top6.png` - 类型评分分布箱线图
8. `platform_mean_rating.png` - 流媒体与传统电视对比

## 功能特点

- 多线程并发抓取，提高效率
- 自动处理网络错误和重试
- 数据完整性验证
- 生成多种统计图表

## 配置说明

在 `start.py` 中可以修改：
- `START_PAGE` / `END_PAGE`: 抓取的页面范围
- `max_workers`: 并发线程数

在 `src/crwal/util.py` 中可以调整：
- `MIN_INTERVAL`: 请求间隔时间
- `TIMEOUT`: 请求超时时间
- `MAX_RETRY`: 重试次数

## 技术栈

- **requests** - HTTP 请求
- **pandas** - 数据处理
- **matplotlib** - 数据可视化
- **tqdm** - 进度显示
- **numpy** - 数值计算

## 作业要求

| 要求 | 完成情况 |
|------|---------|
| 抓取 ≥200 条数据 | ✅ 默认10页约200条 |
| 包含 8 个必需字段 | ✅ Title, First air date, End date, Rating, Genres, Status, Network, Summary |
| 生成 ≥5 幅图表 | ✅ 共8幅图表 |
| 代码规范 | ✅ 模块化设计，有注释 |
| 错误处理 | ✅ 包含重试和异常处理 |

---

**学号**: 2255396  
**姓名**: Yang Hong
