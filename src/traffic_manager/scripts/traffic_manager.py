import pygame
import sys
import time

pygame.init()

# ----------------------------------
# 기본 설정
# ----------------------------------
WIDTH, HEIGHT = 900, 500
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("T-Intersection Traffic Light Simulator")

FONT = pygame.font.SysFont("malgungothic", 20)  # 한글 폰트 (윈도우 기준), 안 되면 기본 사용
CLOCK = pygame.time.Clock()

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
YELLOW = (240, 220, 50)
GREEN = (50, 200, 80)
GREY = (80, 80, 80)
DARKGREY = (40, 40, 40)
BLUE = (70, 140, 220)  # 좌회전 화살표용 테두리

# ----------------------------------
# 신호 Phase 정의
# ----------------------------------
# Phase 구성:
#  P1_MAIN_GREEN     : 메인 양방향 직진 GREEN
#  P1_YELLOW         : 메인 양방향 YELLOW
#  ALL_RED_1         : 전방향 RED
#  P2_MAIN_LEFT      : 메인 왼→오른 좌회전 화살표 GREEN
#  P2_YELLOW         : 좌회전 종료 노랑 (왼→오른만 YELLOW)
#  ALL_RED_2         : 전방향 RED
#  P3_SIDE_GREEN     : 지선 GREEN
#  P3_YELLOW         : 지선 YELLOW
#  ALL_RED_3         : 전방향 RED
#
# 각 Phase 이름, 지속 시간(초)
PHASE_SEQUENCE = [
    ("P1_MAIN_GREEN", 6),
    ("P1_YELLOW", 2),
    ("ALL_RED_1", 0),
    ("P2_MAIN_LEFT", 4),
    ("P2_YELLOW", 2),
    ("ALL_RED_2", 0),
    ("P3_SIDE_GREEN", 5),
    ("P3_YELLOW", 2),
    ("ALL_RED_3", 0),
]

# ----------------------------------
# 신호등 상태 계산
# ----------------------------------
def get_light_states(phase_name):
    """
    각 Phase에서 개별 신호등(메인 왼→오른, 메인 오른→왼, 지선)의
    R/Y/G/좌회전 상태를 반환.
    True = ON, False = OFF
    """
    # 기본: 전부 RED, 나머지 OFF
    state = {
        "M_LR_R": True,  "M_LR_Y": False, "M_LR_G": False, "M_LR_LEFT": False,
        "M_RL_R": True,  "M_RL_Y": False, "M_RL_G": False,
        "S_R": True,     "S_Y": False,    "S_G": False,
    }

    if phase_name == "P1_MAIN_GREEN":
        # 메인 양방향 직진 GREEN
        state.update({
            "M_LR_R": False, "M_LR_G": True,
            "M_RL_R": False, "M_RL_G": True,
            "S_R": True
        })

    elif phase_name == "P1_YELLOW":
        # 메인 양방향 YELLOW
        state.update({
            "M_LR_R": False, "M_LR_Y": True,
            "M_RL_R": False, "M_RL_Y": True,
            "S_R": True
        })

    elif phase_name == "P2_MAIN_LEFT":
        # 메인 왼→오른 좌회전 보호 (좌회전 화살표만 GREEN)
        state.update({
            "M_LR_R": True, "M_LR_LEFT": True,  # 좌회전 화살표 켜짐
            "M_RL_R": True,
            "S_R": True
        })

    elif phase_name == "P2_YELLOW":
        # 좌회전 종료 노랑: 왼→오른에 노랑만 켜기 (직진/좌회전 OFF)
        state.update({
            "M_LR_R": True, "M_LR_Y": True,
            "M_RL_R": True,
            "S_R": True
        })

    elif phase_name == "P3_SIDE_GREEN":
        # 지선 GREEN
        state.update({
            "S_R": False, "S_G": True
        })

    elif phase_name == "P3_YELLOW":
        # 지선 YELLOW
        state.update({
            "S_R": False, "S_Y": True
        })

    # elif phase_name.startswith("ALL_RED"):
    #     # 이미 기본값이 모두 RED라서 그대로 사용
    #     pass

    return state

# ----------------------------------
# 그리기 함수들
# ----------------------------------
def draw_background():
    SCREEN.fill((30, 30, 30))

    # 간단한 T자형 도로 표시 (회색)
    # 메인 도로 (좌↔우)
    pygame.draw.rect(SCREEN, DARKGREY, (0, HEIGHT//2 - 60, WIDTH, 120))
    # 지선 도로 (위에서 내려오는)
    pygame.draw.rect(SCREEN, DARKGREY, (WIDTH//2 - 60, 0, 120, HEIGHT//2 + 60))

    # 라인 (중앙선 느낌)
    pygame.draw.line(SCREEN, GREY, (0, HEIGHT//2), (WIDTH, HEIGHT//2), 2)
    pygame.draw.line(SCREEN, GREY, (WIDTH//2, 0), (WIDTH//2, HEIGHT//2 + 60), 2)


def draw_signal_box(x, y, title, has_left_arrow=False):
    """
    신호등 박스 (틀)와 제목 텍스트
    """
    pygame.draw.rect(SCREEN, BLACK, (x, y, 120, 220), border_radius=10)
    label = FONT.render(title, True, WHITE)
    SCREEN.blit(label, (x, y - 25))

    if has_left_arrow:
        label2 = FONT.render("(Left Arrow 포함)", True, WHITE)
        SCREEN.blit(label2, (x - 10, y + 225))


def draw_light(x, y, color, on):
    """
    원형 신호등 하나 그리기
    """
    radius = 18
    if on:
        pygame.draw.circle(SCREEN, color, (x, y), radius)
    else:
        pygame.draw.circle(SCREEN, GREY, (x, y), radius)
        pygame.draw.circle(SCREEN, color, (x, y), radius, 2)


def draw_arrow_light(x, y, on):
    """
    좌회전 화살표 신호등 (원 안에 화살표 텍스트)
    """
    radius = 18
    if on:
        pygame.draw.circle(SCREEN, GREEN, (x, y), radius)
    else:
        pygame.draw.circle(SCREEN, GREY, (x, y), radius)
        pygame.draw.circle(SCREEN, GREEN, (x, y), radius, 2)

    arrow_text = FONT.render("←", True, WHITE if on else GREEN)
    rect = arrow_text.get_rect(center=(x, y))
    SCREEN.blit(arrow_text, rect)


def draw_all_signals(state, current_phase, phase_remaining):
    """
    state에 따라 세 개의 신호등을 화면에 그림
    """
    # 메인 왼→오른 신호등 위치
    x1, y1 = 100, 150
    draw_signal_box(x1, y1, "메인 왼→오른", has_left_arrow=True)

    # 빨/노/초
    draw_light(x1 + 60, y1 + 40, RED, state["M_LR_R"])
    draw_light(x1 + 60, y1 + 90, YELLOW, state["M_LR_Y"])
    draw_light(x1 + 60, y1 + 140, GREEN, state["M_LR_G"])
    # 좌회전 화살표
    draw_arrow_light(x1 + 60, y1 + 190, state["M_LR_LEFT"])

    # 메인 오른→왼 신호등 위치
    x2, y2 = WIDTH - 220, 150
    draw_signal_box(x2, y2, "메인 오른→왼", has_left_arrow=False)

    draw_light(x2 + 60, y2 + 40, RED, state["M_RL_R"])
    draw_light(x2 + 60, y2 + 90, YELLOW, state.get("M_RL_Y", False))
    draw_light(x2 + 60, y2 + 140, GREEN, state["M_RL_G"])

    # 지선 신호등 위치
    x3, y3 = WIDTH//2 - 60, 40
    draw_signal_box(x3, y3, "지선 (위→아래)", has_left_arrow=False)

    draw_light(x3 + 60, y3 + 40, RED, state["S_R"])
    draw_light(x3 + 60, y3 + 90, YELLOW, state["S_Y"])
    draw_light(x3 + 60, y3 + 140, GREEN, state["S_G"])

    # 현재 Phase 정보 표시
    phase_text = FONT.render(f"현재 Phase: {current_phase}", True, WHITE)
    time_text = FONT.render(f"남은 시간: {phase_remaining:.1f} s", True, WHITE)
    SCREEN.blit(phase_text, (20, HEIGHT - 60))
    SCREEN.blit(time_text, (20, HEIGHT - 30))


# ----------------------------------
# 메인 루프
# ----------------------------------
def main():
    phase_idx = 0
    phase_name, phase_duration = PHASE_SEQUENCE[phase_idx]
    phase_start_time = time.time()

    while True:
        dt = CLOCK.tick(60) / 1000.0  # 초 단위 delta time

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Phase 시간 계산
        elapsed = time.time() - phase_start_time
        remaining = max(0.0, phase_duration - elapsed)

        # Phase 전환
        if elapsed >= phase_duration:
            phase_idx = (phase_idx + 1) % len(PHASE_SEQUENCE)
            phase_name, phase_duration = PHASE_SEQUENCE[phase_idx]
            phase_start_time = time.time()

        # 현재 Phase의 신호 상태 얻기
        state = get_light_states(phase_name)

        # 그리기
        draw_background()
        draw_all_signals(state, phase_name, remaining)

        pygame.display.flip()


if __name__ == "__main__":
    main()
