# MTSC

MTSC 是一个基于 Mooncake Store 和 Transfer Engine（TE）的两阶段 PD KV 传输组件：Decode 先从 Store 加载命中的前缀，再从 Prefill 拉取剩余 KV。

## 组件

- `mtsc.connector.MTSCConnector`：vLLM 外部 KV Connector 入口，P/D 共用同一个实现。
- `mtsc.scheduler.MTSCScheduler`：执行 Store lookup、KV block 分配和请求生命周期管理。
- `mtsc.worker.MTSCWorker`：驱动两阶段加载状态机：Store `[L,A)`，然后 PD `[A,T)`。
- `mtsc.store.StoreIO`：封装 Mooncake Store 的异步 GET/PUT、lookup RPC 和完成通知。
- `mtsc.pd.PDTransfer`：负责 P worker 注册、D/P 拓扑匹配以及 Mooncake TE 数据传输。
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
export MOONCAKE_CONFIG_PATH=/workspace/MTSC/mooncake.json
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
```

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
