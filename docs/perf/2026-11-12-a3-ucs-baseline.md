# A3 单机 1P1D：MTSC vs HiSchedule(UCS) 测试背景、进度与 TODO

> 日期：2026-11-12（打点/实验）～ 11-13（收口）
> 分支：`perf/A3`（MTSC）/ `dev/ssp`（UCS，`UCS-PR/fork/HiSchedule`）
> 本文只记录**实测**结论；推断项都显式标了「假设」。

---

## 0. 结论摘要

| 指标（k468，c10×500，store on，768 GB） | MTSC（`a27542f`） | UCS（读路径修复后） |
|---|---|---|
| TTFT p50 / p90 / p99 / avg (ms) | 1,192 / 5,082 / 7,595 / 2,311 | **1,145 / 4,419 / 5,789 / 2,104** |
| e2e avg / p99 (ms) | 5,024 / 9,948 | **4,803 / 8,301** |
| ITL avg / p99 (ms) | 27.44 / 32.3 | 27.29 / 31.1 |
| req/s · input tok/s | 1.980 · 62.4k | **2.067 · 65.1k** |
| Store 淘汰 · 占用 · PUT | **0** · 531 GB/768 · 181k | **0** · 533 GB/768 · 182k |
| 命中 >7 块 · 命中总量 | 333/500 · 9.90 M tok | 333/500 · 9.76 M tok |

- **I/O 已完全对齐**：一阶段 store load（D 侧 57 vs 60 ms）、store save（10 vs 13 ms）、PD TE 纯传输（写 14 vs 读 29 ms）、ITL、decode 单步（26 ms）——**都不是差的原因**。
- 差的是**两个"等待"**：
  1. **P 的 prefill 尾部**：p50 一样（616 vs 572 ms），**p90 差 1.0 s**（2,672 vs 1,665）；
  2. **D 的"等 KV"饥饿**：MTSC 18% 的 step 空转（UCS 2%）、每步解码 5.39 token（UCS 5.67）→ D 解码墙钟 241 vs 228 s → **整轮 253.9 vs 241.9 s**。
- 在 c10 闭环里总时长差被排队放大：`TTFT_avg ≈ 总时长/(N/c) − OSL×ITL`（MTSC 2.34 s / UCS 2.14 s，与实测 2.31 / 2.10 吻合）。所以 **+4% 的总时长 → p50 +4%、p90 +15%、p99 +31%**。
- **最强假设（未最终验证）**：UCS 的 proxy 在把请求发给 D **之前**会轮询 CP（`/api/v1/pd/early/{id}`，50 ms 一次），只有 P 侧 `publish_pd_early`（P 已 alloc、block_ids 已确定）之后才发 D；MTSC 的 proxy 是 P/D **并行直发**，D 从一到达就挂在等 KV，占着 decode 槽位。见 §4.5 与 §5 P0。

---

## 1. 测试背景

### 1.1 环境

| 项 | 值 |
|---|---|
| 机器 | 180（7.242.105.180），Ascend A3 16×910 64 GB，driver 25.5.1 / CANN 9.0.1 |
| 容器 | `mtsc_test`（host network，镜像 `mtsc_env:efs`），vLLM 0.23.0 / vllm-ascend 0.23.0rc1 |
| 模型 | Qwen3-Coder-30B-A3B-Instruct（BF16，MoE A3B） |
| P（prefill） | TP8 + EP，NPU 0–7，port 8100（UCS 9666），`--enforce-eager` / `cudagraph FULL_DECODE_ONLY` |
| D（decode） | TP8 + EP，NPU 8–15，port 8200（UCS 9667） |
| 共同参数 | `max-model-len 65536`、`max-num-batched-tokens 16384`、`chunked-prefill`、`async-scheduling`、`no-enable-prefix-caching`、`gpu-mem 0.85/0.90` |
| Store | Mooncake master（50051/51051）+ HTTP metadata；`P2PHANDSHAKE`、`protocol=ascend`；`global_segment_size` 48 GiB/客户端 × 16 = **768 GB**，**整轮 0 淘汰** |
| 负载 | aiperf，`--streaming`，concurrency 10，500 请求 |

### 1.2 数据集（k468）

| 指标 | 值 |
|---|---|
| 请求数 / isl | 500 / avg **31,466** tok（p50 31,618、p90 40,175、合计 15.76 M） |
| block / 唯一块 | 128 tok；prompt 总块数 123,161，去重 ≈ 45.7 k（≈530 GB / 4 shard × 2.9 MB）|
| 结构 | **无整条重复**（K=1）；全量共享 **7 块（896 tok）** 的 codex preamble；168 条之后基本是"上一条 + 新回合"的嵌套轨迹 |
| 前缀可复用上限 | **333/500 条可命中 >7 块，中位 196 块，合计 9.91 M tok = 62.9%**（尾部 37% 独有，缓存拿不到）|

### 1.3 Store 配置演变

| 阶段 | `global_segment_size` | 总容量 | 淘汰 | 命中 >7 块 |
|---|---|---|---|---|
| 初始（写放大未修） | 32 GiB | 480 GB | **25 轮 / 199,794 keys / 585 GB** | 139/500 |
| 写索引去重后 | 32 GiB | 480 GB | 4 轮 / 31,327 keys / 91.8 GB | 278/500 |
| 扩容后（当前） | **48 GiB** | **768 GB** | **0** | **333/500（= 上限）** |

---

## 2. 打点体系（归因手段）

| 栈 | 开关 | 事件 |
|---|---|---|
| MTSC | `MTSC_LIFECYCLE_PROFILE_FILE` | `queued`（arrival/queue_ms）、`query_done`（store 查询）、`alloc`、`prefill_done`、`pull_arrive`、`send_permit`、`te_send`（queue/build/write）、`store_load_submit/done`、`store_save`（去重率）、`store_verify`、`stage2_submit/done`、**`step`（tokens/reqs/preempted/step_ms）** |
| UCS | `UCS_LIFECYCLE_PROFILE_FILE` | `request_queued`、`cp_prefix_match`、`connector_match/allocated`、`prefill_start/done`、`pd_ready_publish(_done)`、`mooncake_get_submit/complete`、`te_read_submit/complete`、`pd_ready_received`、`async_save_*`、**`step`（与 MTSC 同字段）** |

两侧都输出 JSONL（单调钟 + 绝对墙钟），分析脚本在 `local/`（不入库）：

```bash
python local/parse_lifecycle.py P.jsonl D.jsonl --summary   # 逐请求阶段表
python local/stage_stats.py   lifecycle.csv                 # p50/p90/p99
python local/master_stats.py  <session>/mooncake_master.log # 写量/淘汰/占用
python local/save_stats.py    <session>/lifecycle_mtsc_P.jsonl  # 写去重率/落盘率
python local/hit_compare.py                                 # 逐请求命中覆盖
python local/timeline.py                                    # 时间线（含口径注意事项）
```

---

## 3. 进度（实验记录）

| # | 实验（run label） | 改动 | TTFT p50 / p90 / p99 / avg | ITL avg | req/s | 结论 |
|---|---|---|---|---|---|---|
| 0 | `lifecycle_mtsc` | 基线（480 G，无写索引） | 4,992 / 8,257 / 9,751 / 4,868 | 27.0 | ≈1.35 | 写放大 + 淘汰致命中崩塌 |
| 1 | `mtsc_dedup` | **写索引去重** | 3,608 / 7,208 / 9,349 / 3,489 | — | — | puts 355,378→182,008；淘汰 25→4 |
| 2 | `mtsc_cap` | **容量 768 GB** | 1,182 / 5,063 / 7,810 / 2,270 | — | 2.01 | 淘汰 0；命中吃满上限（333/500）；p50 −67% |
| 3 | `mtsc_lookup` | lookup socket 池 + 服务端 ROUTER + TTL 缓存 | 1,238 / 5,203 / 8,325 / 2,343 | — | 1.96 | 端到端零收益（缓存命中仅 2%），默认关 |
| 4 | `mtsc_final` | 上述三者的默认配置（串行/无缓存） | 1,238 / 5,136 / 7,994 / 2,351 | 27.04 | 1.982 | 交付基线 |
| A | `mtsc_A` | **构参+写合并为一次 executor hop** | 1,227 / 5,182 / **6,687** / 2,297 | 27.33 | 1.988 | permit 235→45 ms；p99 −890；p50 被吸收 |
| B | `mtsc_B` | `start_load_kv` 内收割 pool + step 打点 | 1,192 / 5,082 / 7,595 / 2,311 | 27.44 | 1.980 | 早收割无害；step 打点揭示 P 空转/D 饥饿 |
| C | `mtsc_sync` | `lookup_async=false` | 1,318 / 5,167 / 7,778 / 2,323 | 27.56 | 1.971 | 查询 37→5 ms、polls 3→1，但整体更差 → 回退 |
| D | `mtsc_wait` | `wait_for_layer_load` 有界等 pool load（0.1 s） | 1,258 / 5,146 / 7,300 / 2,326 | 27.54 | 1.969 | `alloc→load_done` 276→112 ms，端到端无收益 → 默认关 |
| U1 | `ucs_cap768` | UCS 768 GB（读路径未修） | 5 ok / 495 err | 23.7 | 0.027 | 与容量无关，见 §4.6 |
| U2 | `ucs_fix` | **UCS 读路径修复**（关闭 segment 定向读 + 尾部重试 + recompute） | 1,137 / 4,444 / 5,406 / 2,111 | 27.24 | 2.068 | 500/500 恢复，得到可比基线 |
| U3 | `ucs_step` | UCS 加 step 打点 | 1,145 / 4,419 / 5,789 / 2,104 | 27.29 | 2.067 | 打点无干扰；拿到 step 形态 |
| U4 | `lifecycle_ucs4` | UCS **store off** 参照 | 5,614 / 7,458 / 8,284 / 5,753 | 26.44 | 1.182 | 纯 PD 直传，供 store 收益对照 |

---

## 4. 归因分析

### 4.1 已经对齐（不是差异来源）

| 环节 | MTSC | UCS | 说明 |
|---|---|---|---|
| 一阶段 store load（D 侧同口径） | 56.9 / 102.3 ms | 60.0 / 87.4 ms | 一致 |
| 一阶段 store load（P 侧） | 52.5 / 335 ms | 37.2 / 75 ms | 口径不同：UCS 在 `wait_for_layer_load` 内阻塞（≈纯 I/O），MTSC 异步提交、下个 step 收割 |
| store save（一次 PUT） | 10.1 / 21.5 ms | 12.8 / 16.7 ms | 一致 |
| PD TE 纯传输 | 写 14.4 / 47.1 ms | 读 28.7 / 52.8 ms | 一致（MTSC 略优）|
| 命中覆盖 | 196 / 281 块、333/500、9.90 M tok | 196 / 281、333/500、9.76 M tok | 一致 |
| decode 单步 / ITL | 26.0 ms / 27.4 ms | 25.9 ms / 27.3 ms | 一致 |
| 释放语义 | `FinishedWait(store_save, kv_transfer)` 都要等 | `ack_received + save_complete + 无 pending save` 都要等 | 语义一致 |

### 4.2 真正的差异

| 指标 | MTSC | UCS |
|---|---|---|
| P prefill p50 / p90 | 616 / **2,672 ms** | 572 / **1,665 ms** |
| P 在飞请求（alloc→finished）p50 / p90 | 1 / **4** | 1 / **2** |
| 一个请求 prefill 期间经过的活跃 step p50 / p90 | 2 / **6**（慢请求 7） | 1 / **4** |
| 每步解码 token（= 同时解码的请求数） | **5.39** | **5.67** |
| D 空 step 数 | 1,986（4 段长停，最长 1,351） | 199 |
| D 的"到达 → 拿齐 KV" | 912 / **5,013 ms** | **415 / 784 ms** |
| 整轮时长 | 253.9 s | **241.9 s** |

### 4.3 step 形态（P/D 两侧）

| | MTSC P | UCS P | MTSC D | UCS D |
|---|---|---|---|---|
| 总 step | 7,591 | 5,072 | 11,253 | 9,007 |
| 空 step | **6,990（346 串，串长中位 22）** | 4,504（241 串，串长中位 **1**） | 1,986（**4 串**） | 199（2 串） |
| 空转总时长 | **144 s** | 35 s | 24 s | 24 s |
| 活跃 step 耗时 p50 | 247.9 ms | 390.9 ms | 26.0 ms | 25.9 ms |
| tokens/活跃 step p50 | 9,026 | 12,035 | 5 | 5 |
| 有效吞吐 | **46.4k tok/s** | 26.1k tok/s | — | — |

> MTSC 的 P **单步更快、每 token 更快**，但空转成串（346 串 × ~22 步）；UCS 的 P 慢但几乎不空闲。MTSC 的 D 有 4 次长停（300–1,351 步连续空转），对应"10 个槽同时等 KV"。

### 4.4 守恒关系（为什么 p50 怎么优化都不动）

```
TTFT_avg ≈ 总时长/(N/c) − OSL × ITL
MTSC：253.9/50 − 100×0.0274 = 2.34 s（实测 2.31）
UCS ：241.9/50 − 100×0.0273 = 2.14 s（实测 2.10）
```

c10 是闭环：任何"单点延迟"优化都会被队列吸收（实验 1–4、A、C、D 全部印证），**只有提升总吞吐/总时长才会真正改善 TTFT**。

### 4.5 最强假设：proxy 的门控时机（未最终验证）

- **UCS**：P 在 `update_state_after_alloc` 调 `publish_pd_early`（带上 `remote_block_ids`）→ CP 存条目；proxy 每 50 ms 轮询 `/api/v1/pd/early/{id}`，拿到 metadata 后才构造 D 请求（`build_pd_early_decode_params`）并发给 D。**D 收到请求时已被寻址好，不必在 D 引擎里等 KV。**
- **MTSC**：`proxy/pd_proxy.py:_handle` 里 `create_task(_run_prefill(...))` 之后**立刻** `client.post(decode ...)`，P/D 并行直发 → D 从一到达就挂在 `WAITING_FOR_REMOTE_KVS`，占着 decode 槽位（p90 等 5.0 s）。
- 数据吻合：D 空转 18% vs 2%、解码并发 5.39 vs 5.67、整轮 253.9 vs 241.9 s。
- 另一条已测但被**否定**的假设：异步 store lookup 造成批碎片化——同步实验（C）把 `polls` 从 3 降到 1、`query_ms` 37→5 ms，批填充（9,026→9,073 tokens/step）与 TTFT 都没有改善。

### 4.6 UCS 为什么"坏了"（已修复）

- 现象：store-on 时 5/500 成功，`reason=Mooncake get returned failures`（P 2,970 / D 993 次），master 侧 `Get` 计数却**全成功**。
- 根因：CP 给出 `source_segments` 时 UCS 走 `get_from_segments_async`，该路径的失败只在客户端可见；而 `_finish_mooncake_recv` 把"状态数组未覆盖/为负"的尾部块判为失败，叠加默认 `kv_load_failure_policy=fail` 直接判死整条请求。
- 修复（三处）：`ucs_mooncake_use_source_segments` 默认 **False**（改走普通 `batch_get`）；`_finish_mooncake_recv` 对未覆盖尾部**重试一次**；脚本加 `"kv_load_failure_policy": "recompute"`。
- 结果：5/500 → **500/500**，`Mooncake get returned failures` 归零，尾部重试 0 次触发。

---

## 5. TODO

### P0（下一步，证据最硬）
1. **proxy 侧门控实验**：在 `proxy/pd_proxy.py` 加 `MTSC_PROXY_DECODE_DELAY_MS`（默认 0，纯 proxy 侧），扫 0 / 200 / 400 / 600 ms，观察是否收敛到 UCS：
   - D 空 step 数（1,986 → ?）、解码并发（5.39 → ?）、整轮时长（253.9 → ?）、TTFT p90（5.08 → ?）。
   - 若 delay ≈ P 的平均 prefill 后曲线吻合，则确认 §4.5，再实现真门控（等 P 的 alloc/ready 信号，而不是固定延迟）。

### P1（P 的 prefill 尾部）
2. 补 `alloc → 首次被调度` 的精确打点（现在用 `send_permit.wait_ms` 反推，有近似误差），把 p90 的 2,672 ms 拆成"排队 vs 计算"。
3. 若确认是计算/干扰：查 P 的 forward 被哪些线程抢占（save 执行器、lookup 线程、TE 服务），考虑 CPU 亲和/线程数收敛。

### P2（提高解码并发）
4. 若 P0 证实"占槽"是主因，评估 D 侧准入（把"等 KV"的请求暂时让出 slot）或 proxy 分批错峰。

### 维护项
5. UCS 的三处修复需要**回传给 UCS owner**（`cache_connector` 的 `ucs_worker/ucs_config` + 脚本里的 `kv_load_failure_policy`）；本地 `UCS-PR/fork` 目前**未提交**。
6. 打点开关默认全关（`MTSC_LIFECYCLE_PROFILE_FILE` / `UCS_LIFECYCLE_PROFILE_FILE`），压测时只在排查轮开启。
7. `local/`、`scripts/` 是否入库或 `.gitignore`，待定。

---

## 6. 复现

```bash
# MTSC（容量/写索引/step 打点已包含）
bash /root/run_exp.sh mtsc_trace <label> 8000
# UCS（读路径修复后）
bash /root/run_exp.sh ucs_trace  <label> 10826
# UCS store off 参照
bash /root/run_exp.sh ucs_nosave <label> 10826
```

- 归档：`/root/MTSC_TRACE/log/<label>/`（`aiperf.out` + `pd_logs/`）
- aiperf 明细：容器 `/root/MTSC_ENV/log/aiperf/<label>_*/profile_export.jsonl`
- 指标落点：容器 `/root/MTSC_TRACE/log/<label>_metrics.json`

## 附录 A：本分支提交

| commit | 内容 |
|---|---|
| `49201b4` | perf(mtsc): dedup store writes and trace per-request lifecycle |
| `a27542f` | perf(mtsc): move transfer-param build off the sender loop, trace per-step |