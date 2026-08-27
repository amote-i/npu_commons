export IS_DEEPSEEK_V4=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=1
export SGLANG_DSV4_FP4_EXPERTS=False
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
  --model-path /root/.cache/modelscope/hub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp \
  --host 127.0.0.1 --port 8000 \
  --tp-size 16 \
  --trust-remote-code \
  --attention-backend dsv4 \
  --device npu \
  --page-size 128 \
  --quantization modelslim \
  --kv-cache-dtype bfloat16 \
  --watchdog-timeout 9000 \
  --mem-fraction-static 0.7 \
  --max-running-requests 32 \
  --disable-radix-cache \
  --c128-page-size 16
