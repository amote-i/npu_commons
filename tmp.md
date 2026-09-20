# ① 稳定性:同一配置再跑 2 次(加上已有 2 次,共 4 次)
timeout 600 python /home/d30060301/tmp/repro_solve_tril_hang.py --child --device 0 \
  --config '{"name": "grid35424_b9_h32", "B": 9, "T": 7872, "lens": null, "H": 32, "out_dtype": "bf16", "a_dtype": "fp32", "scale": 0.1, "contig": "c", "variant": "plain", "cpp": 1, "num_warps": 4, "num_stages": 3, "repeat": 1, "seed": 100017}'

# ② 边界对:负载几乎相同,仅差 288 个 block(最关键的一组)
#    32544 blocks < 32768 → 预期 OK
timeout 600 python /home/d30060301/tmp/repro_solve_tril_hang.py --child --device 0 \
  --config '{"name": "below_32544", "B": 9, "T": 7232, "lens": null, "H": 32, "out_dtype": "bf16", "a_dtype": "fp32", "scale": 0.1, "contig": "c", "variant": "plain", "cpp": 1, "num_warps": 4, "num_stages": 3, "repeat": 1, "seed": 100017}'
#    32832 blocks > 32768 → 预期挂
timeout 600 python /home/d30060301/tmp/repro_solve_tril_hang.py --child --device 0 \
  --config '{"name": "above_32832", "B": 9, "T": 7296, "lens": null, "H": 32, "out_dtype": "bf16", "a_dtype": "fp32", "scale": 0.1, "contig": "c", "variant": "plain", "cpp": 1, "num_warps": 4, "num_stages": 3, "repeat": 1, "seed": 100017}'

# ③ 同 shape 换 launch 配置(区分"块数调度问题"还是"流水线/编译问题")
#    若 w1/s1 也挂 → 基本排除编译流水线因素,指向调度器/固件
timeout 600 python /home/d30060301/tmp/repro_solve_tril_hang.py --child --device 0 \
  --config '{"name": "grid35424_w1s1", "B": 9, "T": 7872, "lens": null, "H": 32, "out_dtype": "bf16", "a_dtype": "fp32", "scale": 0.1, "contig": "c", "variant": "plain", "cpp": 1, "num_warps": 1, "num_stages": 1, "repeat": 1, "seed": 100017}'

# ④ varlen 同总块数(35424),判断是否与 B 折叠方式无关
timeout 600 python /home/d30060301/tmp/repro_solve_tril_hang.py --device 0 --only grid35424_1seq_h32 --no-stop-on-hit
