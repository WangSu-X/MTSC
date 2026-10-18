# MTSC

MTSC 的 Store key 映射、缓存命中 coordinator 和配置辅助组件维护在 `mtsc/kv_cache_pool.py`，不依赖 vLLM 自带的 Mooncake Connector 包。运行仍需要兼容的 vLLM 核心接口及 Mooncake SDK。迁入组件保留上游版权声明，许可证见 `mtsc/LICENSE-vllm`。

MTSC 是一个基于 Mooncake Store 和 Transfer Engine（TE）的两阶段 PD KV 传输组件：Decode 先从 Store 加载命中的前缀，再从 Prefill 拉取剩余 KV。

TCP 部署要求 P/D 均安装 `mooncake-transfer-engine>=0.3.13.post1`，并且不设置
`MC_TCP_PROTO=1`。此版本使用接收端 ACK，在远端完成 GPU 拷贝后才报告 WRITE
完成。旧版 TCP 仅确认本地 socket 写完，会导致 Decode 读取尚未完整到达的 KV；
MTSC 会在启动时拒绝旧版或强制 legacy 模式。升级时需重启 P/D 的所有 worker：

```bash
python -m pip install 'mooncake-transfer-engine>=0.3.13.post1'
unset MC_TCP_PROTO
```

## 组件

- `mtsc.connector.MTSCConnector`：vLLM 外部 KV Connector 入口，P/D 共用同一个实现。
- `mtsc.scheduler.MTSCScheduler`：执行 Store lookup、KV block 分配和请求生命周期管理。
- `mtsc.worker.MTSCWorker`：驱动两阶段加载状态机：Store `[L,A)`，然后 PD `[A,T)`。
- `mtsc.protocol`：Scheduler/Worker 共用的 metadata，以及 P/D 请求与响应协议。
- `mtsc.kv_cache_pool`：`KVCachePool` 接口及 `MooncakeKVCachePool` 实现，包含 lookup RPC、官方 Store key、内存注册、异步 load/save 和 pending save 合并。
- `mtsc.kv_transfer`：`KVTransfer` 接口及 `MooncakeKVTransfer` 实现，包含 bootstrap、TP/DP 映射、MLA 支持、源会话管理和 TE WRITE。
- `mtsc.utils`：共享设备事件、内存区域和 KV 布局转换工具。
- `proxy.pd_proxy`：同时向 Prefill 和 Decode 发起请求，并向两侧传递同一个 `transfer_id`。

```text
Client
  │
  ▼
Proxy ───────────────► Prefill
  │                       │
  │                       ├── save ──► Mooncake Store
  │                       └── TE write ─────┐
  ▼                                         ▼
Decode ── stage 1: Store GET ──► stage 2: pull remaining KV ──► Decode
```

更完整的状态机和协议说明见 [`docs/design`](./docs/design)。

Scheduler→Worker metadata 使用 `pool_loads`、`pool_saves` 分别描述缓存读写，
每个操作的 token 区间为 `[start_token, end_token)`。`transfer_plans` 描述两阶段
加载边界，`transfer_states` 发布 producer 状态，`finished_waits` 指定释放 blocks
前需要等待的 save/send。Worker 统一跟踪所有 pool load/save，并根据实际加载
前缀启动剩余 KV 的 transfer；等待项只发布一次，完成后清理请求状态。

P↔D RPC 的字段与 vLLM Mooncake 对齐：`req_blocks` 以 D 的 `request_id` 为键，
值为 `(transfer_id, destination_block_groups)`。直传 wire 协议对齐上游
[`da9134bc`](https://github.com/vllm-project/vllm/blob/da9134bc89364465f68aa679564435936d157a14/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)：
请求包含 `kv_block_lens`、layer name/index、group index、shared group 和 row offset；
region 按 layer name 及出现次数匹配，仅复制其所属 group 的 blocks。
响应支持 `FINISH` / `CONTINUE` / `ERROR` / `LAYOUT_MISS` 和 `layout_id`。
D 每次发送完整布局（`layout_id=0`）；P 可缓存上游 D 发布的非零 layout ID，缓存缺失时返回 `LAYOUT_MISS`。
本次协议变更需要同时重启 P/D，不能与旧版 MTSC worker 混用。

直传调度对齐 vLLM MooncakeConnector：D 将同一轮可直传的请求批量提交，P 使用消息队列
和固定发送协程，在一条消息内合并本轮 ready 请求后执行 TE WRITE，并分轮回复完成结果。
连续完整 block 满足两侧地址连续条件时合并传输描述符。`num_workers` 默认 10，
发送协程数为其两倍；不同控制消息分别处理。

`mtsc/` 保持 7 个核心模块（另有包入口 `__init__.py`）：

```text
mtsc/
  __init__.py
  connector.py
  scheduler.py
  worker.py
  protocol.py
  kv_cache_pool.py
  kv_transfer.py
  utils.py
```

两个 Mooncake 类直接实现各自的接口；后端创建与配置校验放在 Worker 的私有工厂函数中。

## 后端与并行支持

Worker 默认创建 `MooncakeKVCachePool` 和 `MooncakeKVTransfer`。可在
`kv_connector_extra_config` 显式设置：

```json
{
  "mtsc_pool_backend": "mooncake",
  "mtsc_transfer_backend": "mooncake"
}
```

当前仅提供 Mooncake 后端，未知名称会在初始化时报错。现有 Mooncake Store
配置文件和 TE 配置参数继续生效。

配置继续由各组件直接从 vLLM 读取。`lookup_async` 支持字符串布尔值，例如 `"false"`、`"off"`；
Store lookup、GET 和 PD 超时必须是有限正数，非法值会在初始化 I/O 资源前报错。
Lookup 默认 10 秒，GET 默认 180 秒；PD 默认使用 vLLM 的
`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`，显式设置 `mtsc_pd_timeout_seconds` 可覆盖它。
Lookup 超时转换为 ZMQ 的有符号 32 位毫秒值，正数小于 1ms 时按 1ms 使用。
`lookup_rpc_port` 用于区分 IPC 路径，0 也是固定标识，不代表自动分配网络端口。

Pool 使用所指定 vLLM Store 的原生 key 格式：
`model@tp_rank:N@pcpN@dcpN@pp_rank:N@group:N@hash`。
`database.process_tokens()` 返回的 `PoolKey` 直接传递给读写路径，在 Mooncake API 边界调用
`to_string()`；lookup 使用相同的 `PoolKey` 序列化。不再添加 MTSC topology namespace 或
`cache_prefix`，原先带 `mtsc-v1-*` 前缀的缓存不再参与命中。
Pool 不转换 TP/PP/PCP/DCP 布局，共享 Store 的实例须保持 model、dtype、block/layout
及 group 语义兼容。MLA 在 DCP=1 时共享 `tp_rank:0` key 并分摊 PUT，物理布局兼容时可跨 TP size 复用。
Transfer 对齐官方的物理 TP 比例切片，要求 `PP=1` 且 P/D TP size 互为整数倍。
MLA 使用官方复制策略；普通注意力按有效 KV head 数处理 TP 分片及 GQA 复制 rank，
支持 P/D 跨越 KV head 数边界的组合。padding/NVFP4 等不支持的 head 重分片会在 WRITE 前明确失败并回退。
DP 仍通过 Proxy 指定的 engine/DP 选择副本。协议校验 layer/group 对应关系与 payload 长度；
model、dtype、block/layout 以及 PCP/DCP 配置仍须由部署保持兼容。

MTSC 保留两阶段加载的空目标确认、失败 block 回报，以及抢占/关闭时的在途 I/O 等待。
接收超时与官方一致，为 P ready 超时加 60 秒。P 明确返回失败时可回退重算；
超时或断连后无法确认 WRITE 已停止时，会报告致命 worker 错误并保留注册内存，等待进程退出。
单元测试使用模拟 TE 和真实 ZMQ 控制通道，性能及设备行为仍需在 GPU/NPU 上验证。

## 配置示例

### 1. Mooncake Store

Store 从 `MOONCAKE_CONFIG_PATH` 读取 `protocol`、`device_name` 等配置。
直传 TE 的 `protocol` 和 `device_name` 按字段分别采用以下优先级：

1. `kv_connector_extra_config` 中的 `mooncake_protocol` / `device_name`。
2. `MOONCAKE_CONFIG_PATH` 指向的 JSON 中的 `protocol` / `device_name`。
3. 默认值：`rdma` / 空字符串。

`null` 视为未配置；显式设置 `device_name: ""` 可以清除 JSON 的网卡白名单。
需要读取 JSON 时，文件不存在或内容无效会使初始化失败，不会静默切换协议。
例如 JSON 设置 `protocol: "tcp"` 时，Store 和直传默认均使用 TCP；在 extra config
显式设置 `mooncake_protocol: "rdma"` 可只覆盖直传。共享 JSON 默认值是 MTSC 的扩展，
仍保留官方的 extra config 入口；直传 metadata 模式保持 `P2PHANDSHAKE`。

创建 Mooncake 配置文件，例如 `/workspace/MTSC/mooncake.json`：

```json
{
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "master_server_address": "127.0.0.1:50051",
  "protocol": "rdma",
  "device_name": "",
  "global_segment_size": "4GB",
  "local_buffer_size": "4GB"
}
```

启动 P/D 前设置：

```bash
export PYTHONPATH=/workspace/MTSC:/workspace/vllm-0.26.0
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=/workspace/MTSC/mooncake.json
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
```

P/D 必须使用相同的 `PYTHONHASHSEED`。未设置时，vLLM 会为每个进程随机生成
KV block hash 的起始值，导致相同前缀无法跨进程命中 Store。

做 PD 与 native 的严格输出一致性测试时，可在 P/D 和基线进程启动前均设置：

```bash
export VLLM_BATCH_INVARIANT=1
```

vLLM 默认的批次及 prefill 分段可能改变 BF16 数值结果，即使 `temperature=0`
且 seed 相同，也可能让接近并列概率的 token 发生分叉。当前 vLLM 0.23 的
Qwen2.5-0.5B 测试已在纯 vLLM 中复现该现象，启用上述模式后原 PD 用例的文本
和返回 logprobs 均匹配基线。cold/warm prefix-cache 的计算路径仍应分别对照；
该模式的性能影响需单独评估。功能说明见
[vLLM Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)。

### 2. Prefill

Prefill 使用 `kv_producer`：

```json
{
  "kv_connector": "MTSCConnector",
  "kv_connector_module_path": "mtsc.connector",
  "kv_role": "kv_producer",
  "engine_id": "prefill-0",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "load_async": true,
    "lookup_async": true,
    "mooncake_protocol": "rdma",
    "device_name": "",
    "mtsc_pd_timeout_seconds": 180,
    "mtsc_store_lookup_timeout_seconds": 10,
    "mtsc_store_get_timeout_seconds": 180
  }
}
```

示例启动命令：

```bash
vllm serve /path/to/model \
  --port 8100 \
  --kv-transfer-config '{
    "kv_connector":"MTSCConnector",
    "kv_connector_module_path":"mtsc.connector",
    "kv_role":"kv_producer",
    "engine_id":"prefill-0",
    "kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{
      "load_async":true,
      "lookup_async":true,
      "mooncake_protocol":"rdma",
      "mtsc_pd_timeout_seconds":180,
      "mtsc_store_lookup_timeout_seconds":10,
      "mtsc_store_get_timeout_seconds":180
    }
  }'
```

### 3. Decode

Decode 使用 `kv_consumer`，并配置独立的 `engine_id`：

```json
{
  "kv_connector": "MTSCConnector",
  "kv_connector_module_path": "mtsc.connector",
  "kv_role": "kv_consumer",
  "engine_id": "decode-0",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "load_async": true,
    "lookup_async": true,
    "mooncake_protocol": "rdma",
    "device_name": "",
    "mtsc_pd_timeout_seconds": 180,
    "mtsc_store_lookup_timeout_seconds": 10,
    "mtsc_store_get_timeout_seconds": 180,
    "mtsc_decode_save": true
  }
}
```

示例启动命令：

```bash
vllm serve /path/to/model \
  --port 8200 \
  --kv-transfer-config '{
    "kv_connector":"MTSCConnector",
    "kv_connector_module_path":"mtsc.connector",
    "kv_role":"kv_consumer",
    "engine_id":"decode-0",
    "kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{
      "load_async":true,
      "lookup_async":true,
      "mooncake_protocol":"rdma",
      "mtsc_pd_timeout_seconds":180,
      "mtsc_store_lookup_timeout_seconds":10,
      "mtsc_store_get_timeout_seconds":180,
      "mtsc_decode_save":true
    }
  }'
```

`kv_load_failure_policy` 必须设置为 `recompute`。当 Store 或 PD transfer 失败时，vLLM 会从首个失败 block 开始本地重算。

### 4. Proxy

```bash
python -m proxy.pd_proxy \
  --prefill http://127.0.0.1:8100,prefill-0,http://127.0.0.1:8998,0 \
  --decode http://127.0.0.1:8200 \
  --metrics-file /workspace/logs/mtsc/requests.jsonl \
  --metrics-queue-size 4096 \
  --first-token-timeout 180 \
  --port 8000
```

`--prefill` 的格式为：

```text
API_URL,ENGINE_ID,BOOTSTRAP_ADDR[,DP_RANK]
```

Proxy 会并发请求 P/D，不执行重试；任一后端失败会直接返回错误。每个请求结束后，会向 `--metrics-file` 写入一条 JSONL 记录。

使用 internal DP 时，Proxy 的 `ENGINE_ID` 必须与 bootstrap `/query` 返回的每个
副本的 ID 一致。例如 vLLM 0.23 会将配置中的 `prefill-0` 改为
`prefill-0_dp0`、`prefill-0_dp1`，因此 DP=2 的两个 Prefill 路由应分别配置为：

```bash
--prefill http://127.0.0.1:8100,prefill-0_dp0,http://127.0.0.1:8998,0 \
--prefill http://127.0.0.1:8100,prefill-0_dp1,http://127.0.0.1:8998,1
```

错误的 ID 会导致 PD pull 被拒绝，随后由 `recompute` 策略本地重算；仅检查
HTTP 200 或生成文本不能确认 PD 传输成功。P/D 同机启动时，也应分别配置
不同的 `kv_connector_extra_config.lookup_rpc_port`，避免 Store lookup IPC 路径冲突。

Store lookup 超时会按 miss 处理。Mooncake 同步 GET 没有安全取消接口，因此 GET 超时后 MTSC 会继续 fence 底层调用，待其终止后将该请求的 Store blocks 标为无效并回退到 P 拉取或本地重算，避免迟到写覆盖已复用的 KV block。

## Ascend

Ascend 环境需要使用开启 `USE_ASCEND_DIRECT=ON`、包含 CANN/ADXL 支持的 Mooncake，并将 Store 和 direct-PD 的传输协议改为 `ascend`：

```json
{
  "protocol": "ascend",
  "device_name": ""
}
```

```json
{
  "kv_connector_extra_config": {
    "mooncake_protocol": "ascend",
    "device_name": ""
  }
}
```

MTSC 支持 `torch.npu.Event`、分离 K/V tensor 和整数倍异构 TP。
普通 NHD KV 按 token 描述传输 region，避免把多个 rank 的 KV 拼成错误的 token/head 顺序；
物理 HND KV 可直接按 head 切片。NZ KV 的异构 head 重分片会明确失败并回退本地重算，
同构布局按原始字节传输。Store 按原始字节保存和恢复 KV，不执行接收后重排；
P/D 须使用兼容的物理布局。Decode Store save 由 `mtsc_decode_save` 控制。

### 开发验证范围

本地 vLLM 核心接口测试使用 v0.23.0。`tests/reference_mooncake.py` 固定保存上游
`da9134bc` 的 wire schema，协议兼容测试不依赖本地安装版本的旧 schema。
CPU 地址复制回归覆盖 NPU NHD/HND、双向异构 TP、GQA、独立/共享 group、packed layer 和 Store GET。
真实 GPU/RDMA 与 Ascend/NZ 集成仍需在对应硬件环境验证。
