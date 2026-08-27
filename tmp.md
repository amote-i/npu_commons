MODEL_PATH=Qwen/Qwen3-0.6B

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --host 127.0.0.1 \
  --port 6688 \
  --device npu \
  --trust-remote-code \
  --enable-http2 \
  --http2-max-concurrent-streams 2

拉起后观测参数生效（单条 h2c 连接）：

# 1) 直接观测：SETTINGS 帧通告值应为 2

curl -v --http2-prior-knowledge http://127.0.0.1:6688/get_model_info 2>&1 | grep SETTINGS_MAX_CONCURRENT_STREAMS

# 2) 行为观测：单连接并发流超过 2 的请求被 RST_STREAM 拒绝（需 pip install h2）

python observe_http2_limit.py 6688 2   # 之前生成的观测脚本，预期 refused >= 3

如需更大并发（如默认 200 或压测场景），把 --http2-max-concurrent-streams 2 改为目标值即可，其余命令不变。
