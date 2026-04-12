# ingest 模块说明

`ingest` 模块负责把外部 Bilibili 数据源变成后续提取流程可用的评论列表，并维护增量处理所需的水位线。

对应代码目录：

- [src/brain/ingest/reader.py](src/brain/ingest/reader.py)
- [src/brain/ingest/cleaner.py](src/brain/ingest/cleaner.py)
- [src/brain/ingest/state.py](src/brain/ingest/state.py)

## 1. 模块职责

当前 `ingest` 拆成三部分：

1. `reader`：从 SQLite 读取视频列表和评论
2. `cleaner`：过滤无效评论并做精确去重
3. `state`：读写水位线，支持增量处理

## 2. 代码位置

### 2.1 `reader.py`

- [src/brain/ingest/reader.py:8-15](src/brain/ingest/reader.py#L8-L15) `BilibiliReader.__init__` / `_connect`
- [src/brain/ingest/reader.py:17-25](src/brain/ingest/reader.py#L17-L25) `list_videos()`
- [src/brain/ingest/reader.py:27-45](src/brain/ingest/reader.py#L27-L45) `read_comments()`

### 2.2 `cleaner.py`

- [src/brain/ingest/cleaner.py:7-12](src/brain/ingest/cleaner.py#L7-L12) `_PURE_EMOJI_RE`
- [src/brain/ingest/cleaner.py:15-21](src/brain/ingest/cleaner.py#L15-L21) `_has_text_content()`
- [src/brain/ingest/cleaner.py:24-35](src/brain/ingest/cleaner.py#L24-L35) `clean_comments()`

### 2.3 `state.py`

- [src/brain/ingest/state.py:6-19](src/brain/ingest/state.py#L6-L19) `WatermarkState`

## 3. 核心对象 / 函数

### 3.1 `BilibiliReader`

#### `list_videos()`

执行 SQL：

```sql
SELECT bvid, title FROM videos WHERE crawl_status = 'completed' ORDER BY bvid
```

特点：

- 只处理 `crawl_status = 'completed'` 的视频
- 结果按 `bvid` 排序
- 返回值是 `list[dict]`，不是 dataclass

#### `read_comments(bvid)`

执行 SQL：

```sql
SELECT rpid, bvid, uid, uname, message, ctime, root, parent
FROM comments
WHERE bvid = ?
ORDER BY ctime
```

特点：

- 每条记录被转成 `CleanedComment`
- 保留 `uname`、`root`、`parent`，供后续构造评论对
- 保留顺序为 `ctime` 升序
- 不在这里做清洗或去重

### 3.2 `clean_comments(comments)`

该函数负责基础清洗，逻辑在 [src/brain/ingest/cleaner.py:24-35](src/brain/ingest/cleaner.py#L24-L35)。

处理规则：

1. `strip()` 去掉首尾空白
2. 长度小于 `MIN_COMMENT_LENGTH` 则过滤
3. 若整条评论匹配 `_PURE_EMOJI_RE`，则过滤
4. 对 `msg` 做精确去重：完全相同的消息只保留第一次出现的那条

注意：

- 去重键是清洗后的完整字符串
- 不做模糊归一化
- 不按用户维度去重
- 不按骨架去重

### 3.3 `WatermarkState`

`WatermarkState` 用 JSON 文件保存每个 source 的处理进度。

方法：

- `get_watermark(source)`
- `set_watermark(source, value)`

存储格式是一个简单字典，例如：

```json
{
  "bilibili": "BV1abc"
}
```

## 4. 输入与输出

### 4.1 `reader`

输入：

- SQLite 数据库路径
- 可选的 `bvid`

输出：

- `list_videos()` → `list[dict]`
- `read_comments()` → `list[CleanedComment]`

### 4.2 `cleaner`

输入：

- `list[CleanedComment]`

输出：

- 清洗后的 `list[CleanedComment]`

### 4.3 `state`

输入：

- `source`
- `value`
- `state.json` 文件内容

输出：

- 水位线字符串
- 更新后的 `state.json`

## 5. 执行流程

### 5.1 单视频评论摄入流程

```text
给定 bvid
    ↓
reader.read_comments(bvid)
    ↓
得到原始 CleanedComment 列表
    ↓
cleaner.clean_comments(comments)
    ↓
过滤短评论 / 纯表情 / 完全重复文本
    ↓
输出清洗后评论列表
```

### 5.2 增量处理流程

`scripts/run_pipeline.py` 中对 `WatermarkState` 的使用见 [scripts/run_pipeline.py:43-47](scripts/run_pipeline.py#L43-L47) 与 [scripts/run_pipeline.py:101-105](scripts/run_pipeline.py#L101-L105)。

流程：

1. 读取 `state.get_watermark("bilibili")`
2. 若存在，则仅处理 `bvid > watermark` 的视频
3. 处理完成后，将最后一个视频的 `bvid` 写回水位线

这说明当前增量策略的单位是：

- **视频级增量**，不是评论级增量

## 6. 依赖关系

- `reader` 依赖 `sqlite3` 和 `CleanedComment`
- `cleaner` 依赖 `MIN_COMMENT_LENGTH` 配置
- `state` 依赖本地 JSON 文件
- `scripts/run_pipeline.py` 直接依赖整个 `ingest` 模块链路

## 7. 当前限制 / 已知偏差

1. `list_videos()` 只看 `crawl_status = 'completed'`，无法表达部分完成状态
2. 增量判断依赖 `bvid` 字符串排序，不是时间戳或数据库水位线
3. `clean_comments()` 只做精确文本去重，不处理轻微变体
4. `reader` 和 `cleaner` 解耦，但没有统一的高层 ingest service 封装
5. `state.py` 使用本地 JSON 文件，没有锁，也没有并发保护

## 8. 测试覆盖

测试位于 [tests/test_ingest.py](tests/test_ingest.py)。

覆盖点包括：

- `BilibiliReader.list_videos()`
- `BilibiliReader.read_comments()`
- `clean_comments()` 的短文本过滤、纯表情过滤、精确去重
- `WatermarkState` 的读写与持久化
