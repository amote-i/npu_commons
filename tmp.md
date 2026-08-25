python -m sglang.test.run_eval \
    --eval-name gsm8k \
    --api completion \
    --max-tokens 512 \
    --temperature 0 \
    --port 8234 \
    --num-examples 200

python -m sglang.test.run_eval \
    --eval-name mmlu \
    --api chat \
    --max-tokens 1024 \
    --temperature 0 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --port 8234 \
    --num-examples 500

python -m sglang.test.run_eval \
    --eval-name humaneval \
    --api chat \
    --temperature 0 \
    --max-tokens 1024 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --port 8234 \
    --num-examples 50

python -m sglang.test.run_eval \
    --eval-name humaneval \
    --api chat \
    --temperature 0.8 \
    --max-tokens 1024 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --port 8234

pip install "sgl-eval@git+https://github.com/sgl-project/sgl-eval.git@6690895609dcbc5df1e7b00dd57c9502b868ec4d"
