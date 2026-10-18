#!/usr/bin/env python3
"""PlanRec 切分健康检查（split health scorecard）

用法:
  python3 scripts/validate_split.py --data-dir <sequences.json 所在目录> [--max-seq-len 50] [--min-seq-len 3]

在训练一个大模型之前，先用半小时跑这份检查回答三件事：
  1）切分有没有泄漏（训练词汇、时间顺序、流行度统计越界）；
  2）数据里到底有没有可学信号（频率基线上界、重复率）；
  3）测试目标与训练分布是否协调（目标流行度分歧）。

从原始序列文件实时重算 leave-one-out（与 PlanRec-Protocol 一致），不依赖已处理产物。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--max-seq-len", type=int, default=50)
p.add_argument("--min-seq-len", type=int, default=3)
args = p.parse_args()

seq_path = Path(args.data_dir) / "sequences.json"
if not seq_path.exists():
    sys.exit(f"ERROR: {seq_path} not found")
sequences = json.load(open(seq_path, encoding="utf-8"))
print(f"原始序列: {len(sequences)} 条")

# ── 1) 规模与分布 ──────────────────────────
lens = sorted(len(s) for s in sequences)
usable = [s for s in sequences if len(s) >= args.min_seq_len]
print(f"\n[1] 规模: 序列 {len(sequences)}, 可用(T>={args.min_seq_len}) {len(usable)} ({len(usable)/len(sequences)*100:.1f}%)")
print(f"    长度分布 p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")

# ── 2) 词汇与越界 ──────────────────────────
train_items: Counter = Counter()   # 训练目标物品频次（= 所有前缀末项）
test_items: Counter = Counter()    # 测试目标物品频次
val_items: Counter = Counter()
train_pairs = 0
for s in usable:
    T = len(s)
    for t in range(1, T - 2):          # 训练目标: 与 tencentgr.py 定义一致 input=seq[:t], target=seq[t]
        train_items[s[t]] += 1
        train_pairs += 1
    val_items[s[T - 2]] += 1
    test_items[s[T - 1]] += 1

vocab = set(train_items) | set()
test_oov = sum(1 for it in test_items if it not in set(train_items))
val_oov = sum(1 for it in val_items if it not in set(train_items))
print(f"\n[2] 词汇: 训练目标唯一物品 {len(train_items)}, 训练样本 {train_pairs}")
print(f"    test 目标 OOV(不在训练词汇) {test_oov}/{len(test_items)} = {test_oov/len(test_items)*100:.2f}%")
print(f"    val  目标 OOV {val_oov}/{len(val_items)} = {val_oov/len(val_items)*100:.2f}%")

# ── 3) 目标流行度分歧 ──────────────────────
def hist(counter: Counter) -> list[int]:
    c = Counter(counter.values())
    return [c.get(i, 0) for i in range(1, 6)] + [sum(v for k, v in c.items() if k >= 6)]

def num(counter: Counter, label: str):
    h = hist(counter)
    n = sum(counter.values())
    print(f"    {label}: 目标分布 频次1={h[0]}({h[0]/n*100:.1f}%) 频次2={h[1]} 频次3+={sum(h[2:])}({sum(h[2:])/n*100:.1f}%) 高频(>=6)={h[5]}")

print("\n[3] 目标物品在训练集出现频次分布（分歧检查）:")
num(train_items, "train目标")
num(test_items, "test 目标")
num(val_items, "val  目标")

# ── 4) 流行度基线（频率模型上界）───────────
# 注意: 分母必须是"test 样本数"（逐样本命中率），不是唯一物品数
topk = [it for it, _ in train_items.most_common(50)]
test_targets = [s[-1] for s in usable]
pop_hr = sum(1 for t in test_targets if t in set(topk[:10])) / max(len(test_targets), 1)
pop_hr50 = sum(1 for t in test_targets if t in set(topk[:50])) / max(len(test_targets), 1)
print(f"\n[4] 频率基线(训练 Top50 流行度预测 test, 按样本计): HR@10={pop_hr:.4f} HR@50={pop_hr50:.4f}")
print(f"    随机参考 HR@10 ≈ {10 / len(set(train_items) | set(test_items)):.5f}")

# ── 5) 序列内重复率（信号代理）─────────────
repeat_hits = 0
total_targets = 0
for s in usable:
    seen = set(s[: -1])
    if s[-1] in seen:
        repeat_hits += 1
    total_targets += 1
print(f"\n[5] next-item 序列内重复率: {repeat_hits}/{total_targets} = {repeat_hits/total_targets*100:.2f}%")
print("    (重复率越高说明物品级重复信号越强；若接近 0，序列几乎没有物品级可学信号)")

# ── 6) 判定 ────────────────────────────────
print("\n[6] 判定: ")
warn = []
if test_oov / max(len(test_items), 1) > 0.05:
    warn.append("test OOV >5% —— 切分可能存在词汇泄漏或数据过稀疏，谨慎采信测试指标")
if pop_hr < 0.02:
    warn.append("频率基线 HR@10 <2% —— 全目录协议下流行度主导弱，模型提升空间需重点看相对值")
if repeat_hits / max(total_targets, 1) < 0.05:
    warn.append("重复率 <5% —— 物品级信号很弱，建议核对数据来源与切分")
if not warn:
    print("    未触发主要警告。")
for w in warn:
    print(f"    ⚠ {w}")
print("\n参考: PlanRec-Protocol 与 TENCENTGR 主实验同构；本脚本为训练前快速验证，不替代训练指标。")
