# MTSC

MTSC 的 Store key 映射、缓存命中 coordinator 和配置辅助组件维护在 `mtsc/kv_cache_pool.py`，不依赖 vLLM 自带的 Mooncake Connector 包。运行仍需要兼容的 vLLM 核心接口及 Mooncake SDK。迁入组件保留上游版权声明，许可证见 `mtsc/LICENSE-vllm`。

MTSC 是一个基于 Mooncake Store 和 Transfer Engine（TE）的两阶段 PD KV 传输组件：Decode 先从 Store 加载命中的前缀，再从 Prefill 拉取剩余 KV。

TCP 部署要求 P/D 均安装 `mooncake-transfer-engine>=0.3.13.post1`，并且不设置
`MC_TCP_PROTO=1`。此版本使用接收端 ACK，在远端完成 GPU 拷贝后才报告 WRITE
完成。旧版 TCP 仅确认本地 socket 写完，会导致 Decode 读取尚未完整到达的 KV；
MTSC 会在启动时拒绝旧版或强制 legacy 模式，并用新的 TCP schema 拒绝旧版
MTSC 对端。升级时需重启 P/D 的所有 worker：

```bash
python -m pip install 'mooncake-transfer-engine>=0.3.13.post1'
unset MC_TCP_PROTO
```

## 组件

- `mtsc.connector.MTSCConnector`：vLLM 外部 KV Connector 入口，P/D 共用同一个实现。
- `mtsc.scheduler.MTSCScheduler`：执行 Store lookup、KV block 分配和请求生命周期管理。
- `mtsc.worker.MTSCWorker`：驱动两阶段加载状态机：Store `[L,A)`，然后 PD `[A,T)`。
- `mtsc.protocol`：Scheduler/Worker 共用的 metadata，以及 P/D 请求与响应协议。
- `mtsc.kv_cache_pool`：`KVCachePool` 接口及 `MooncakeKVCachePool` 实现，包含 lookup RPC、namespace、内存注册、异步 load/save 和 pending save 合并。
- `mtsc.kv_transfer`：`KVTransfer` 接口及 `MooncakeKVTransfer` 实现，包含 bootstrap、TP/PP/DP 映射、MLA 支持、源会话管理和 TE WRITE。
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

P↔D RPC 使用 `KVTransferRequest` / `KVTransferResponse`，协议版本为 6（TCP 为 7）。
请求映射、完成/失败列表和 region coverage 均以 `transfer_id` 为键，
Worker 向 Scheduler 报告完成时仍使用 `request_id`。P 和 D 需要同步升级；
旧协议会在 schema 校验时被拒绝。

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

配置继续由各组件直接从 vLLM 读取。`lookup_async` 和 NPU 的
`additional_config.enable_kv_nz` 支持字符串布尔值，例如 `"false"`、`"off"`；
`mtsc_store_topology_namespace` 缺省或为 `null` 时均默认开启。
Store lookup、GET 和 PD 超时必须是有限正数，非法值会在初始化 I/O 资源前报错。
Lookup 默认 10 秒，GET 默认 180 秒；PD 默认使用 vLLM 的
`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`，显式设置 `mtsc_pd_timeout_seconds` 可覆盖它。
Lookup 超时转换为 ZMQ 的有符号 32 位毫秒值，正数小于 1ms 时按 1ms 使用。
`lookup_rpc_port` 用于区分 IPC 路径，0 也是固定标识，不代表自动分配网络端口。

Pool 支持普通 KV 与 MLA，但不转换 TP/PP/PCP/DCP 布局；默认使用 topology namespace
隔离不兼容缓存。MLA 在同一 namespace 内跨 TP ranks 共享 key，并分摊完整 chunk 的 PUT。
MLA 部署在满足 `PCP=1` 和 `DCP=1` 时，不同 TP size 使用统一的 namespace，允许
P (TP=8) 和 D (TP=1) 共享 Store 对象，无需 Decode 回写 prompt KV。普通注意力
及 PCP>1 或 DCP>1 的 MLA 继续按 TP size 隔离。model、dtype、block/layout、
group 语义、PP、PCP/DCP 配置须保持兼容。
Transfer 支持整数倍异构 TP、按 layer 交集匹配的异构 PP，以及不同 DP size/rank 的指定副本路由。
MHA/GQA 按唯一 KV 分片切片，MLA 复制完整 latent KV 并去除重复发送者。model、dtype、
block/layout、group 语义及 PCP/DCP 配置仍须兼容。

Transfer 控制协议使用 `topology_version=6`（TCP 为 7），包含双方 engine/DP 身份、目标
block 容量及批量请求的分轮结果。P/D 应一起升级；旧协议会被拒绝。

## 配置示例

### 1. Mooncake Store

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

MTSC 支持 `torch.npu.Event`、分离 K/V tensor、整数倍异构 TP 重排以及 `enable_kv_nz`。启用 NZ layout 时，MTSC 会关闭 Decode Store save，避免将 Decode 的物理布局写入 Prefill 共用的 Store key。
