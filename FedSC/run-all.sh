#!/usr/bin/env bash
set -e               # dừng nếu có lệnh lỗi
set -o pipefail

# ==== HYPER-PARAMS DÙNG CHUNG ====
NUM_CLIENTS=10        # paper = 10
ROUNDS=80             # paper (CIFAR) = 80
LOCAL_EPOCHS=5
BATCH=64
LR=0.01
LAMBDA_RPCL=1.0
LAMBDA_CPDR=0.2
TAU=0.05
DEVICE="cuda"         # hoặc "cpu" nếu không có GPU

# ==== 1. IID baseline (CIFAR-10) ====
python fedsc_scenarios.py \
  --scenario iid --dataset cifar10 --num_clients $NUM_CLIENTS \
  --rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --batch_size $BATCH \
  --lr $LR --lambda_rpcl $LAMBDA_RPCL --lambda_cpdr $LAMBDA_CPDR \
  --tau $TAU --device $DEVICE

# ==== 2. NID1 (Dirichlet α = 0.2) ====
python fedsc_scenarios.py \
  --scenario nid1 --alpha 0.2 --dataset cifar10 --num_clients $NUM_CLIENTS \
  --rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --batch_size $BATCH \
  --lr $LR --lambda_rpcl $LAMBDA_RPCL --lambda_cpdr $LAMBDA_CPDR \
  --tau $TAU --device $DEVICE

# ==== 3. NID2 (5 client chỉ giữ 2 lớp, 1 client đủ lớp) ====
python fedsc_scenarios.py \
  --scenario nid2 --dataset cifar10 --num_clients $NUM_CLIENTS \
  --rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --batch_size $BATCH \
  --lr $LR --lambda_rpcl $LAMBDA_RPCL --lambda_cpdr $LAMBDA_CPDR \
  --tau $TAU --device $DEVICE

# ==== 4. Long-tailed CIFAR-100 với ρ ∈ {10,50,100,200} ====
for RHO in 10 50 100 200; do
  python fedsc_scenarios.py \
    --scenario longtail --rho $RHO --alpha 0.1 \
    --dataset cifar100 --num_clients $NUM_CLIENTS \
    --rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --batch_size $BATCH \
    --lr $LR --lambda_rpcl $LAMBDA_RPCL --lambda_cpdr $LAMBDA_CPDR \
    --tau $TAU --device $DEVICE
done

# ==== 5. Few-shot FC100 (5-way 5-shot, paper) ====
python fedsc_scenarios.py \
  --scenario fewshot --k_shot 5 --dataset fc100 --num_clients $NUM_CLIENTS \
  --rounds 40 --local_epochs 5 --batch_size 32 \\
  --lr $LR --lambda_rpcl $LAMBDA_RPCL --lambda_cpdr $LAMBDA_CPDR \
  --tau $TAU --device $DEVICE
