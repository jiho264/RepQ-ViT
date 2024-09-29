#!/bin/bash

# 모델 리스트
models=("vit_small" "vit_base" "deit_tiny" "deit_small" "deit_base" "swin_tiny" "swin_small")

# 실행할 스크립트

# 모델별로 반복 실행
for model in "${models[@]}"
do
    # 로그 파일 이름 설정
    log_file="${model}_org_16.txt"
    
    # 스크립트 실행 및 로그 파일에 출력 저장
    echo "Running experiment with model: $model"
    python -u test_quant.py --model $model > $log_file 2>&1
    
    # 로그 저장 확인 메시지
    echo "Logs saved to $log_file"
done
