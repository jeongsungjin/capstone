# Vehicle uplink status
int32 vehicle_id          # 차량 번호/ID
float32 voltage           # 전압 (V), 소수점 2자리까지 사용 권장
float32 heading_diff      # Δψ (rad, wrap to [-pi,pi]) since last uplink
float32 heading_dt        # Δt (sec) matched to heading_diff integration window
uint32 heading_seq        # IMU 측 시퀀스 번호 (연속성/드롭 추적용)

