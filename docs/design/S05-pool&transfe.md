# MTSC KVCachePool 与 KVTransfer 设计

> 上位文档：[S00-整体设计](./S00-整体设计.md)  
> 正式接口：[kv_cache_pool.py](../../mtsc/kv_cache_pool.py)、[kv_transfer.py](../../mtsc/kv_transfer.py)

本文定义外部 KV 能力的抽象契约，现已由 `MooncakeKVCachePool(KVCachePool)` 和
`MooncakeKVTransfer(KVTransfer)` 实现。接口与对应 Mooncake 实现放在同一个模块中，
具体类持有 namespace、内存注册、bootstrap 和底层 I/O；Worker 通过 ABC 提交事件和消费结果。

## 1. 背景

MTSC 已通过 scheduler/worker 串联远程缓存和 P-D 直传：D 先加载远程缓存的命中前缀，再请求 P 补齐剩余 KV；P、D 均可保存计算产生的 KV。目前两条路径直接依赖 Mooncake Store 和 Transfer Engine，接口中还混合了存储任务完成与 request 结束后的 block 释放逻辑。

在保留现有两阶段加载、增量保存和安全回收行为的前提下，需要把 external 能力拆成两个独立接口：

- `KVCachePool`：远程 KV 缓存的包装层，后端可选 Mooncake Store、MongoDB、FileSystem 等。
- `KVTransfer`：worker 间已有 KV 的直接传输，当前采用 D 发起请求、P WRITE D。

目标是：接口只传必要元数据，不复制 KV tensor；调用方不维护 task ID 映射；队列、线程、Future 和底层 I/O 由实现选择。request 调度、block 分配和最终释放仍由 Connector 与 vLLM 管理。

## 2. 总体设计

### 2.1 核心特性

1. **存储与传输分离**：Pool 不处理 P-D 路由，Transfer 不处理缓存命中和持久化。
2. **按 request 管理**：结果直接携带本地 `request_id`；`transfer_id` 只用于跨 P/D 匹配。
3. **异步提交、非阻塞轮询**：load/save/send/recv 返回 `None`，通过 `poll()` 收取结果，即时完成也走同一通道。
4. **增量 save 合并**：同一 request 可多次保存，合并安全的 pending 范围，减少排队任务和存储调用。
5. **完成与成功分离**：失败也必须等相关内存访问停止后才报告完成；无效目标 block 单独上报。
6. **单写者与安全回收**：Pool load 终止后才启动 P WRITE；最终 block 释放需要满足 request 的所有依赖。

### 2.2 架构设计

```mermaid
flowchart LR
    X[PDProxy] -->|并发请求及 transfer_id| S[各引擎 MTSCScheduler]
    S -->|lookup| LC[Lookup Client]
    LC -->|RPC 查询| P[Worker KVCachePool]
    S -->|每步 metadata| W[MTSCWorker]
    W -->|load / save / poll| P
    P --> B[(远程存储后端)]
    W -->|send / recv / poll| T[KVTransfer]
    T -->|worker 发现| BS[Bootstrap]
    T <-->|控制请求与完成通知| RT[对端 KVTransfer]
    RT -->|P WRITE D 数据路径| T
    W -->|完成及 invalid_block_ids| O[vLLM Worker Output]
    O --> S
```

图中 Worker/Transfer 节点代表本地与对端实例，实际 KV 数据方向固定为 P 到 D。scheduler 的查询通过轻量 lookup client 访问 worker Pool，不在 scheduler 初始化完整的 GPU I/O 对象。

### 2.3 组件职责

| 组件 | 职责 |
|---|---|
| `PDProxy` | 选择 P/D，向两侧传入同一 `transfer_id`，向 D 提供 P 的 bootstrap、engine 和 DP 身份 |
| `MTSCConnector` | 适配 vLLM hooks，将 scheduler/worker 的完成和错误状态接入 vLLM |
| `MTSCScheduler` | 查询可用前缀，绑定分配的 blocks，生成增量 metadata，决定 request 结束时的释放依赖 |
| Lookup Client | 包装查询 RPC 与可选的异步等待，隔离 scheduler 和 worker 的进程边界 |
| `MTSCWorker` | 编排 Pool-first 两阶段加载，收取结果，处理 fallback，聚合 request 的 block 释放条件 |
| `KVCachePool` | 注册本地 KV 内存，查询远程缓存，执行 load/save，管理 pending 合并及存储任务的安全终态 |
| `KVTransfer` | 注册传输内存与 endpoint，发现 P workers，匹配 send/recv，校验布局与覆盖，执行直传 |

### 2.4 类图

```mermaid
classDiagram
    class MTSCWorker
    class KVCachePool {
        <<abstract>>
        register(kv_caches)
        lookup(token_count, hashes) int
        load(LoadEvent)
        save(SaveEvent)
        poll() PoolPollResult
        take_errors() set
        preempt(request_id)
        close()
    }
    class KVTransfer {
        <<abstract>>
        register(kv_caches)
        prepare(request_id, transfer_id)
        cancel(request_id, transfer_id)
        send(SendEvent)
        recv(RecvEvent)
        poll() TransferPollResult
        take_errors() set
        preempt(request_id)
        close()
    }
    class PoolPollResult {
        loads: LoadResult[]
        saves: SaveResult[]
    }
    class TransferPollResult {
        sends: TransferResult[]
        recvs: TransferResult[]
    }
    MTSCWorker --> KVCachePool : 存储任务
    MTSCWorker --> KVTransfer : P2P 任务
    KVCachePool ..> LoadEvent
    KVCachePool ..> SaveEvent
    KVCachePool ..> PoolPollResult
    PoolPollResult o-- LoadResult
    PoolPollResult o-- SaveResult
    KVTransfer ..> SendEvent
    KVTransfer ..> RecvEvent
    KVTransfer ..> TransferPollResult
    TransferPollResult o-- TransferResult
```

事件和结果均为简单 dataclass；后端实现继承相应 ABC。传输状态机与 Future 不对调用方暴露。

### 2.5 核心方法

| 方法 | 核心行为 |
|---|---|
| `register()` | 注册本地 KV tensor；group、layout、topology 等静态信息由后端初始化配置提供 |
| Pool `lookup()` | 只查询远程可用前缀，不分配 block、不搬运数据 |
| Pool `load()` | 将指定 token 范围加载到已分配 block，通常每个 allocation 生命周期一个逻辑任务 |
| Pool `save()` | 接收增量保存范围，同 request 的 pending 工作在安全条件下合并 |
| Transfer `prepare()` / `cancel()` | 登记尚未 ready 的 P 会话；永久取消并拒绝迟到请求 |
| Transfer `send()` | P 发布已经 ready 的源 block 表，等待 D 请求后发送 |
| Transfer `recv()` | D 发现目标 P workers，提交目标 block 表，等待 P 写入和终态通知 |
| `poll()` / `take_errors()` | 收取安全终态与无效目标 block；Worker 决定 fallback 或最终上报 |
| `preempt()` / `close()` | 停止或等待相关 I/O，内存访问终止后才允许复用 block 或释放资源 |

### 2.6 E2E 时序图

```mermaid
sequenceDiagram
    participant PS as P Scheduler
    participant PW as P Worker
    participant PP as P Pool
    participant PT as P Transfer
    participant DS as D Scheduler
    participant DW as D Worker
    participant DP as D Pool
    participant DT as D Transfer
    participant BS as Bootstrap

    Note over PS,DS: Proxy 已向两侧传入同一 transfer_id
    par P 计算与保存
        PS->>PW: 增量 SaveEvent
        PW->>PP: save，等待 ready 信号后保存
        PS->>PW: request_finished，源 KV ready及释放依赖
        PW->>PT: send，发布源 block 表
    and D 加载
        DS->>DP: 经 lookup client 查询候选前缀 H
        DP-->>DS: 命中长度
        DS->>DW: 已分配 blocks及加载计划 L/H/T
        DW->>DP: load [L,H)
        DW->>DP: poll + take_errors
        DP-->>DW: 安全终态，实际可用前缀 A
        DW->>DT: recv，目标后缀 blocks
        DT->>BS: 查询指定 engine/DP 的 P workers
        BS-->>DT: TP/PP worker 地址
        DT->>PT: transfer_id + 目标 blocks及内存信息
    end
    Note over PT,DT: send 和 D 请求到达顺序不限；两者齐备后传输
    PT->>DT: WRITE 剩余 KV [A,T)
    PT-->>DT: 完成或失败通知
    DW->>DT: poll + take_errors
    DT-->>DW: 接收终态及无效 blocks
    DW-->>DS: finished_recving及最终 invalid_block_ids
    Note over DS: 完整加载后 decode；无效 blocks交给 vLLM 重算
    PW->>PP: poll，等待全部 save终止
    PP-->>PW: SaveResult
    PW->>PT: poll，等待全部目标发送终止
    PT-->>PW: send TransferResult
    PW-->>PS: 所有释放依赖满足后 finished_sending
```

P、D 均可独立 lookup/load/save。图中突出 D 两阶段加载和 P 源 block 释放；D decode 期间的增量 save 复用同一 Pool 协议。

## 3. 详细设计

### 3.1 数据契约

`BlockIds = tuple[tuple[int, ...], ...]`：外层按 KV group 排列，内层是有序的本地物理 block ID。事件使用不可变元数据快照，KV tensor 在注册后保持有效。

| 结构 | 必要字段与含义 |
|---|---|
| `LoadEvent` | `request_id`、完整 `block_ids` / `block_hashes`、token 范围 `[start_load,end_load)` |
| `SaveEvent` | `request_id`、完整 `block_ids` / `block_hashes`、token 范围 `[start_save,end_save)`、可选 `ready_event` / `prompt_tokens`（用于 hybrid 保存 mask） |
| `LoadResult` | `request_id`、实际可用前缀终点 `loaded_tokens`、可选 `error` |
| `SaveResult` | `request_id`、本轮聚合保存的可选 `error` |
| `SendEvent` | 本地 `request_id`、跨侧 `transfer_id`、完整源 `block_ids` |
| `RecvEvent` | 本地 `request_id`、`transfer_id`、仅待接收的目标 `block_ids`、`bootstrap_addr`、`remote_engine_id`、`remote_dp_rank` |
| `TransferResult` | 本地 `request_id`、可选 `error`；由 sends/recvs 列表区分方向 |

Pool 根据 block hash 和静态布局配置生成存储 key；Transfer 不传 block hash，直接根据会话和物理 block 定位数据。`request_id` 不能在旧任务、结果或 block 错误尚未清理时复用。

### 3.2 Lookup 与单次 Load

`lookup(token_count, hashes) -> int` 返回远程可用的逻辑前缀长度，miss 返回 0，后端异常抛出。scheduler 侧的 client 可异步包装查询，并将查询失败按 miss 降级。lookup 结果是候选值，不能保证数据在 load 时仍存在。

一次 load 可以包含多个 group/chunk 子任务，推荐内部维护 `request_id -> load future`，由该 future 聚合全部子任务。`poll()` 必须等全部子任务终止，包括失败任务，保证不会继续写入目标内存，再返回一个 `LoadResult`。

Worker 根据实际终点 `A = loaded_tokens` 决定 PD 后缀，不能使用候选终点 H 提前启动 WRITE：

```text
0 <= L <= A <= H <= T
Pool load: [L,H)    实际可用前缀截至 A
P2P WRITE: [A,T)    覆盖剩余后缀及需要重写的 blocks
```

`loaded_tokens` 包含已有本地前缀 L；有效性按各 group 的缓存规则解释。加载失败 blocks 在同轮 `take_errors()` 收取，Worker 过滤被 PD 成功恢复的 blocks，只向 scheduler 上报最终无效部分。

### 3.3 增量 Save 与合并

同一 request 随 prefill/decode 进展可提交多个 save。推荐内部维护 `request_id -> save state`，包含 running 工作、pending events 和累计错误。常见连续范围只需一个 running 和一个合并后的 pending；不满足合并条件的快照保留独立工作。

```text
running: [0,64)
pending: [64,96) + [96,128) -> [64,128)
```

合并只修改 pending 元数据，不能改变执行中的快照。范围必须连续或重叠，最新 block/hash 表必须覆盖合并范围且保留原 token 到源 block 的对应关系；等待的 ready 信号必须覆盖全部源数据。不能跨抢占或 block 复用合并。只有完整且 ready 的缓存 chunk 被保存。

`SaveResult` 覆盖上一次结果收取后接受的全部 save。返回条件是 **所有相关工作终止、没有 pending event、没有源内存读取**，不能只检查现有 futures。较早的错误不能被后续成功覆盖；结果收取前新到达的 save 会扩展待完成工作，`poll()` 必须重新检查，避免返回过期完成状态。

request 继续运行时可多次收到 `SaveResult`，之后也可继续 save。Pool 不需要 `finish(request_id)`；request 结束与最终 block 释放由 Worker 聚合。

### 3.4 D 发起请求、P WRITE

1. P 在 KV 可被传输层安全读取后调用 `send()`，发布 ready 的完整源 block 表；此调用不等待接收方或 I/O。
2. D 调用 `recv()`，后端经 bootstrap 查找指定 engine/DP 的 workers，派生 TP/PP 配对及目标内存描述，发送控制请求。
3. P 等源发布与 D 请求齐备，按每组目标 block 数量，选取源表等长后缀执行 WRITE。
4. P 完成写入后通知 D；P 聚合全部预期目标的终态，D 聚合全部必要 worker 的响应与覆盖校验。

源和目标必须对应同一前缀终点，group/block 布局兼容；当前不支持任意中间区间。send 与 D 控制请求谁先到都可等待匹配。D 成功结果必须包含必要的本地布局转换与设备同步。目标为空时不搬运 KV，但仍需会话确认，使 P 能结束源会话；这一已有行为需在后端适配时保留。

### 3.5 完成、错误与 Block 生命周期

`poll()` 非阻塞且结果只收取一次；终态表示内存访问已停止，成功与否由结果判断。`take_errors()` 非阻塞取出并清空已轮询失败的物理目标 block ID。它只处理 load/recv 的无效目标，不因 save/send 失败而失效本地源 KV。

Worker 的 `_send` 表示 request 结束后的释放依赖，D 仅保存时也需要它。`_save_issued` 表示生成过保存工作，不等于当前正在执行 save。

```text
finished_sending = request 已结束
                  AND 所有已提交 save 终止（如需要）
                  AND P-D send 终止（如需要）
```

中间 `SaveResult` 只更新 Worker 的保存状态，不能直接进入最终 `finished_sending`。新 save 到达后重新标记保存未完成。request 结束时没有 outstanding save 的情况，也应利用已有状态完成释放判断。

`preempt()` 可以阻塞：取消 pending 工作，fence 无法安全取消的运行任务，清理未收取结果和错误后才允许复用 block。
P 尚未 `send()` 时没有发布源内存，抢占保留早到的 D 等待者，可在恢复后用原 `transfer_id` 发布新的 ready blocks。
永久取消必须通过 `cancel(request_id, transfer_id)` 结束该会话；已发布 send/recv 的尝试则退休 ID，后续重试使用新 ID。
`prepare()` 将本地 request 与未 ready 的会话关联，也允许 `cancel()` 在尚未 prepare 时退休指定 ID。D 已发布目标内存时必须等待远端写停止；timeout、取消本地 Future 或忽略迟到响应均不能单独证明安全。`close()` 同样先 fence，再释放注册内存与后端资源。

## 4. Mooncake 实现与并行映射

### 4.1 实现入口

| 功能 | 当前代码路径 |
|---|---|
| 正式 ABC、事件与结果 | `mtsc/kv_cache_pool.py`、`mtsc/kv_transfer.py` |
| Pool 生命周期与 I/O | `mtsc/kv_cache_pool.py`：lookup RPC、namespace、注册、GET/PUT、load 聚合、pending save 合并、错误与 fence |
| Transfer 生命周期与 I/O | `mtsc/kv_transfer.py`：bootstrap、注册、region 对齐与 WRITE、prepare/send/recv/cancel、去重、退休 ID 与 fence |
| Worker 编排 | `mtsc/worker.py`：消费 metadata、Pool-first、以 loaded_tokens 固化 A、聚合 request 结束后的释放依赖 |
| Backend 配置 | `mtsc/worker.py` 私有工厂：`mtsc_pool_backend` / `mtsc_transfer_backend`，默认且当前仅支持 `mooncake` |
| 共享协议 | `mtsc/protocol.py`：Scheduler/Worker metadata 与 P/D wire 结构 |
| 设备布局转换 | `mtsc/utils.py`：共享设备事件、注册区域、NPU 转换与有效 KV 分片数 |
| KV 字节分片映射 | `mtsc/kv_transfer.py`：`kv_slice_plan()`，处理 TP 分片与复制 rank |

scheduler-worker metadata 保留现有 wire 结构，Worker 将其转换为不可变事件快照。
Pool 不接收 request-end 信息，Worker 的 `_save_pending` 与 `_send` 分别管理当前保存工作及最终释放依赖。
两个 Mooncake 类直接实现 ABC；旧的任务生命周期、完成 API 和兼容导出已移除。

### 4.2 Pool 并行与 MLA

Pool 不执行跨 TP/PP/PCP/DCP 布局转换；默认 topology namespace 包含这些维度及 model、dtype、layout、group schema。
DP 是副本身份，不进入 KV 分片 key 的布局坐标，兼容副本可以共享缓存。

MLA 在 TP 间复制 latent KV，同一 namespace 内使用有效 KV head 数 1 和共享 TP key；
各 rank 按 chunk 分摊 PUT，每个 rank 都能加载完整副本。暂不移除 namespace 中的 TP size，
因此 MLA 也不额外启用跨 TP size 的 Pool 复用。

### 4.3 Transfer TP/PP/DP

物理 TP rank 用于发现与控制会话，唯一 KV 分片用于数据切片：

```text
MHA/GQA: effective_tp = min(tp_size, total_kv_heads)
MLA:     effective_tp = 1
replica_count = tp_size / effective_tp
```

P/D TP size 保持整数倍要求。每个 D rank 的 P peer 集合中，同一个 KV 分片只选择一个发送者。
P 分片较细时拼接到 D 的对应 byte offset；P 分片较粗时取 D 所需切片。
覆盖次数按唯一分片计算，避免 GQA 在 TP 超过 KV head 数时漏算或重复计算；MLA 始终复制完整 page。

PP size 相同时联系配对 rank，不同时联系对应 TP 的全部 P PP workers，再按全局 layer 身份及 region 顺序取交集。
兼容 group/layout 仍是前提，不支持任意 group 语义或 block 大小转换。
DP 使用 Proxy 指定的 engine/DP 身份选择 P 副本，不进行跨副本聚合或隐式请求迁移。
PCP/DCP size 进入 Transfer schema 并要求一致。

Transfer 协议版本为 2，携带双方 engine/DP 身份、目标容量及内存描述。每个源会话绑定一个 D 副本/拓扑，
按唯一 `(engine, DP, TP, PP)` 目标聚合终态；重复请求只等待原任务，不再次 WRITE 或重复计数。
同一目标修改 destination 描述会被拒绝。退休 ID 保留到后端 close，阻止迟到请求再次读取已释放源 blocks。

### 4.4 Fence 与清理

任何接收 peer 失败后，仍等待其余已发布目标地址的 peer 终态，才报告 recv 失败。
源超时不能结束 active WRITE；空目标仍完成控制握手。
P shutdown 保留控制 socket，直到在途 handler 发出终态响应，再关闭 listener 和注册内存。
Pool timeout 标记失败但等待同步 GET 返回。pending save 可取消，running save 必须停止源读取后才能复用 blocks。

## 5. 验证与后续工作

`tests/test_backends.py` 覆盖具体后端的 save 合并、ready 信号、累计错误、增量结果、部分 load、timeout、
抢占、空 recv、迟到请求、目标去重、指定 DP 身份、多 peer fence，以及模拟 TE 字节写入的异构 TP/PP/MLA。
`tests/test_mtsc.py` 保留 namespace、MLA key、设备注册、coverage、lookup、Proxy 和 legacy helper 回归测试。
这些测试不代表真实 Mooncake/GPU E2E 已通过。

后续验证：真实 GPU/Mooncake 的同构及异构 TP/PP E2E、MLA 模型与 Ascend NZ 实机验证；
替代 Pool 后端可在相同 ABC 下另行实现。非整数倍 TP 和跨 Pool 布局复用不在本版范围。
