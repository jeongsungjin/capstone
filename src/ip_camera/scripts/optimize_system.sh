#!/bin/bash

echo "=== 시스템 최적화 스크립트 ==="

# 1. 네트워크 버퍼 크기 최적화
echo "네트워크 버퍼 최적화..."
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.rmem_default=65536
sudo sysctl -w net.core.wmem_default=65536

# 2. TCP 최적화
echo "TCP 최적화..."
sudo sysctl -w net.ipv4.tcp_rmem="4096 65536 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# 3. CPU 스케줄링 최적화
echo "CPU 스케줄링 최적화..."
sudo sysctl -w kernel.sched_rt_runtime_us=-1

# 4. 메모리 최적화
echo "메모리 최적화..."
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.dirty_ratio=15
sudo sysctl -w vm.dirty_background_ratio=5

echo "최적화 완료!"
echo "현재 설정 확인:"
echo "rmem_max: $(sysctl net.core.rmem_max)"
echo "wmem_max: $(sysctl net.core.wmem_max)"
echo "tcp_congestion_control: $(sysctl net.ipv4.tcp_congestion_control)"
