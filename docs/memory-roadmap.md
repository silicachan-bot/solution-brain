# 记忆模块实现路线

## 1. 总体设计原则

- brain 作为纯库，不依赖 bot；bot 单向依赖 brain
- 两个子项目通过 uv path dependency 连接，直接 `import brain`，无 HTTP 服务层
- 对话模型：kimi k2.5（支持 tool calling，中文质量好）

---

## 2. 记忆层级

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: 对话历史 (ephemeral, in-memory)           │
│  session_id → deque[messages]，进程内，不持久化       │
├─────────────────────────────────────────────────────┤
│  Layer 2: 自我记忆 (persistent, LLM 可写)            │
│  硅酱的人格、规则、认知、偏好                          │
├─────────────────────────────────────────────────────┤
│  Layer 3: 用户记忆 (persistent, 对话后提取)           │
│  每个 QQ 号对应的认知 + 用户间关系网络                │
├─────────────────────────────────────────────────────┤
│  Layer 4: 语言模式库 (persistent, 离线写入)           │
│  网络梗/说话风格，从 Bilibili 离线提取，对话中只读      │
└─────────────────────────────────────────────────────┘
```

### Layer 1：对话历史

- 存储位置：bot 进程内存，`dict[session_id, deque[messages]]`
- session_id：群聊为 `{group_id}_{user_id}`，私聊为 `private_{user_id}`
- 不持久化，bot 重启清空
- 会话超时（15 分钟无消息）时触发用户记忆提取

### Layer 2：自我记忆（灵魂）

- 存储位置：`solution-brain/data/soul.json`
- **不使用向量检索，始终全量注入 system prompt**
  - 原因：人格规则是触发式逻辑（"当某事发生时如何反应"），向量检索只能匹配话题，无法命中未被话题触发的规则
- JSON 结构：`intro`（基本设定）+ `entries[]`（各类条目）
- 分类：`personality` / `rule` / `preference` / `knowledge`
- Token 预算：~800 token，控制文件体积
- **条目必须以第一人称书写**（以"我"开头），`Soul.add_entry()` 在写入时强制校验
- **写入触发**：
  - LLM 调用 `update_self_memory` tool（对话中随时）
  - 受信任用户（白名单 QQ 号）发出指令时，LLM 自主判断写入

### Layer 3：用户记忆

- 存储位置：LanceDB `user_memory` 表
- 按 `user_id` 过滤后语义检索，取相关条目注入
- 分类：`fact` / `personality` / `preference` / `relationship`
- 关系类条目额外含 `related_user_id` 字段，构成用户关系网络
- **写入触发**：
  - 对话结束后，LLM 异步提取对话中体现的用户信息（`extractor.py`）
  - LLM 调用 `note_about_user` tool（对话中实时记录）

### Layer 4：语言模式库

- 存储位置：LanceDB `patterns` 表（双路向量：`vec_template` + `vec_semantic`）
- 来源：Bilibili 评论离线提取 pipeline
- 对话中只读，按语义检索 top-k 注入菜单
- 用途：补充网络梗知识，弥补模型不能实时上网的缺陷，不等同于人格记忆

---

## 3. system prompt 结构

```
## 我是谁
{intro}

## 我的性格
- {entries...}

## 我的规则
- {entries...}

...（自我记忆全量）

【关于 {username}】
- {user_memory top-k}

【可用语言模式】
{pattern_menu}

如果用户对你的说话方式有反馈、或你觉得有什么值得记住的，
可以调用 update_self_memory 更新你自己的记忆。
```

Token 估算：自我记忆 ~800 + 用户记忆 ~300 + 模式菜单 ~400 + 对话历史 ~1500 = **~3000 token**，对 kimi k2.5 无压力。

---

## 4. LLM 可用工具

| 工具 | 用途 | 何时调用 |
|---|---|---|
| `inspect_pattern(pattern_id)` | 查看语言模式详情 | 想用某个模式前 |
| `update_self_memory(entry, category)` | 写入自我认知 | 用户反馈 / 需要记住某事 |
| `note_about_user(user_id, content, category)` | 记录用户信息 | 对话中发现值得记的用户信息 |

tool calling loop 上限 3 轮，防止模型循环调用。

---

## 5. 单次对话流程

```
收到消息
  ↓
preprocessor.clean_text()
  ↓
session.get_history(session_id)
  ↓
┌─ 并行 ─────────────────────────────────┐
│  retrieve_patterns(query)               │  ← Layer 4
│  user_memory.retrieve(user_id, query)   │  ← Layer 3
└─────────────────────────────────────────┘
  ↓
self_identity.load()                       ← Layer 2（全量）
  ↓
assemble_system_prompt(self_doc, user_mems, patterns)
  ↓
llm.chat_with_tools(messages, tools, tool_handler)
  → tool loop（inspect_pattern / update_self_memory / note_about_user）
  ↓
session.append(session_id, exchange)
  ↓
bot.send(reply)
  ↓（异步，不阻塞回复）
on_session_end → extractor.extract_user_facts → user_memory.save_batch
```

---

## 6. 数据目录结构

```
solution-brain/data/
  soul.json        # Layer 2：bot 灵魂（JSON，全量加载）
  patterns/        # Layer 4：语言模式库（LanceDB）
  users/           # Layer 3：用户记忆（LanceDB，待实现）
  state.json       # pipeline 水位线，非记忆层
```

## 7. 文件结构

### solution-brain

```
src/brain/
  tools.py                # ✅ 已实现：统一 tool 注册入口（get_tools / dispatch）
  memory/
    __init__.py
    soul.py               # ✅ 已实现：Soul，JSON 存储，全量加载，第一人称校验
    tools.py              # ✅ 已实现：update_self_memory tool 定义 + handler
    user_memory.py        # ⬜ 待实现：LanceDB，按 user_id 过滤 + 语义检索
    extractor.py          # ⬜ 待实现：对话后 LLM 提取用户事实
  compose/
    assembler.py          # ✅ 已更新：接受 soul + patterns 参数
    tools.py              # ✅ 已实现：inspect_pattern tool 定义 + handler
  prompts/
    compose_system.txt    # ✅ 已更新：注入灵魂记忆 + 工具说明
  config.py               # ✅ 已更新：SOUL_FILE / PATTERNS_DIR / USERS_DIR
```

### chat-core

```
services/
  llm_client.py           # ✅ 已更新：新增 chat_with_tools()，支持 tool loop
plugins/llm_chat/
  __init__.py             # ✅ 已更新：初始化 Soul，改为 kimi 配置
  handler.py              # ✅ 已更新：动态 system prompt，通过 brain.tools 统一调度
  session.py              # ⬜ 待实现：对话历史 deque + 超时管理
```

---

## 8. 待实现清单

- [ ] `brain/memory/user_memory.py` — UserMemoryDB（LanceDB）
- [ ] `brain/memory/extractor.py` — 对话后异步提取用户事实
- [ ] `brain/tools.py` — 补充 `note_about_user` tool（接入 user_memory 后）
- [ ] `chat-core/plugins/llm_chat/session.py` — 对话历史管理 + 会话超时触发
- [ ] `compose/assembler.py` — 接入用户记忆（当前只有灵魂记忆 + 模式）
- [ ] 白名单机制 — 受信任用户可指令式修改灵魂记忆
- [ ] 灵魂记忆压缩 — Token 超出预算时触发 LLM 压缩
