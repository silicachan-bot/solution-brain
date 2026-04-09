# compose 模块

`compose` 模块负责把检索出来的 `PatternCard` 组织成可注入的文本和工具接口，供后续聊天系统消费。

对应代码：
- [src/brain/compose/menu.py](../../src/brain/compose/menu.py)
- [src/brain/compose/assembler.py](../../src/brain/compose/assembler.py)
- [src/brain/compose/tools.py](../../src/brain/compose/tools.py)
- [src/brain/compose/templates/system.txt](../../src/brain/compose/templates/system.txt)

## 1. 模块职责

`compose` 当前负责：

1. 把 `PatternCard` 列表压缩成菜单文本
2. 把菜单插入 system prompt 模板
3. 定义 `inspect_pattern` 工具 schema
4. 根据 `pattern_id` 返回模式详情文本

它不负责：
- 检索哪些模式
- 调用真实模型完成对话
- 管理 tool-calling 会话状态

## 2. 代码位置

- 菜单生成：[src/brain/compose/menu.py](../../src/brain/compose/menu.py)
- prompt 组装：[src/brain/compose/assembler.py](../../src/brain/compose/assembler.py)
- 工具定义与处理：[src/brain/compose/tools.py](../../src/brain/compose/tools.py)
- 模板文件：[src/brain/compose/templates/system.txt](../../src/brain/compose/templates/system.txt)

## 3. 核心对象 / 函数

### 3.1 `build_menu()`

定义在 [src/brain/compose/menu.py:5-12](../../src/brain/compose/menu.py#L5-L12)。

行为：
- 输入 `list[PatternCard]`
- 输出编号列表字符串
- 每一行格式为：

```text
1. [pat-xxxx] "模板" — 描述前30字
```

如果没有模式，则返回：

```text
当前没有可用的语言模式。
```

### 3.2 `assemble_system_prompt()`

定义在 [src/brain/compose/assembler.py:12-16](../../src/brain/compose/assembler.py#L12-L16)。

行为：
1. 调用 `build_menu(patterns)`
2. 读取 [src/brain/compose/templates/system.txt](../../src/brain/compose/templates/system.txt)
3. 用 Jinja2 模板渲染 `menu`
4. 返回最终 system prompt 字符串

模板当前内容非常短，核心意思是：
- 模型可以先调用 `inspect_pattern`
- 再决定是否使用某种语言模式
- 也可以完全不使用任何模式

模板内容见 [src/brain/compose/templates/system.txt:0-3](../../src/brain/compose/templates/system.txt#L1-L4)。

### 3.3 `get_tool_definition()`

定义在 [src/brain/compose/tools.py:5-23](../../src/brain/compose/tools.py#L5-L23)。

返回一个 OpenAI/Claude 风格的 function tool 定义，当前工具名固定为：

```text
inspect_pattern
```

参数只有一个：
- `pattern_id: string`

### 3.4 `handle_inspect_pattern()`

定义在 [src/brain/compose/tools.py:25-36](../../src/brain/compose/tools.py#L25-L36)。

行为：
- 调用 `db.get(pattern_id)` 读取卡片
- 如果不存在，返回“找不到”提示
- 如果存在，返回一段格式化文本，格式为：

  ```text
  模板：{card.template}
  描述：{card.description}
  例句：
    - {ex1}
    - {ex2}
  ```

它当前返回的是纯文本，不是结构化 JSON。

## 4. 输入与输出

### 4.1 菜单层

输入：
- `list[PatternCard]`

输出：
- 菜单字符串

### 4.2 prompt 层

输入：
- `list[PatternCard]`
- `system.txt` 模板

输出：
- 最终 system prompt 文本

### 4.3 工具层

输入：
- `PatternDB`
- `pattern_id`

输出：
- tool schema 字典
- 或模式详情文本

## 5. 执行流程

当前 `compose` 的流程很短：

```text
retrieve_patterns(...) 返回 PatternCard 列表
  -> build_menu(...)
  -> assemble_system_prompt(...)
  -> get_tool_definition()
  -> 运行时如果模型请求 inspect_pattern
  -> handle_inspect_pattern(db, pattern_id)
```

展开后：

1. 检索层给出一组相关卡片
2. `menu.py` 把卡片压成列表菜单
3. `assembler.py` 把菜单注入模板
4. 外部调用方把 tool schema 一起交给模型
5. 模型如需查看详情，再通过 `handle_inspect_pattern()` 取文本说明

## 6. 依赖关系

依赖：
- [src/brain/models.py](../../src/brain/models.py) 中的 `PatternCard`
- [src/brain/store/pattern_db.py](../../src/brain/store/pattern_db.py) 中的 `PatternDB`
- `jinja2`

被这些模块或场景使用：
- 测试文件 [tests/test_compose.py](../../tests/test_compose.py)
- 当前查看器里直接用了 `build_menu()` 预览菜单，见 [scripts/streamlit_patterns.py:119-120](../../scripts/streamlit_patterns.py#L119-L120)
- 未来真实对话集成会复用这层接口

## 7. 当前限制 / 已知偏差

1. 当前 compose 只负责“文本组装”，不负责真实模型调用。
   - 所以它更像一个 prompt/tool payload 构造层。

2. `inspect_pattern` 返回纯文本，不返回结构化字段。
   - 对模型是友好的，但对程序化消费不够强。

3. `assemble_system_prompt()` 每次都会读模板文件。
   - 当前规模下没问题，但没有缓存。

4. 模板目前非常短，没有分层策略或 token 预算控制。
   - 现在实现目标是先把最小闭环搭出来。

## 8. 测试覆盖

测试位于 [tests/test_compose.py](../../tests/test_compose.py)。

覆盖点包括：
- `build_menu()`
- `get_tool_definition()`
- `handle_inspect_pattern()`
- `assemble_system_prompt()`
