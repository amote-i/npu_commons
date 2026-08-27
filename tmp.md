source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_PATH=Qwen/Qwen3-0.6B

python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --host 127.0.0.1 --port 6688 \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --tp-size 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --prefill-decode-interval 16 \
  --max-running-requests 16

关键参数说明:

- --enable-dp-attention --dp-size 2 --tp-size 2:开启 DP attention(tp 组按 dp 切分,每个 die 一个 DP rank),这是该参数设计的目标场景,跨 rank 用 batch.is_extend_in_batch 同步节奏 (scheduler.py:1230-1236)。
- --prefill-decode-interval 16:每执行 1 个 prefill/extend 批次后,后续 16 个调度轮次跳过 prefill 只跑 decode (scheduler.py:1216-1221,3186-3187)。设为 0 即禁用(默认)。

生效验证:

1. 启动日志中 Server Args: 会打印 prefill_decode_interval=16。
2. 发请求后观察调度行为:把 interval 调大(如 1000000)再并发发多个请求,可看到第 1 个请求 prefill 后,后续请求的 prefill 被持续推迟、已有请求继续 decode(排队数增长、无新 prefill 执行)。
