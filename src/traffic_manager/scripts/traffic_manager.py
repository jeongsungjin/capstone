import pygame
import sys
import time

pygame.init()

# ----------------------------------
# 기본 설정
# ----------------------------------
WIDTH, HEIGHT = 1240, 900
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Way T-Intersection Traffic Simulator")

FONT = pygame.font.SysFont("malgungothic", 18)
CLOCK = pygame.time.Clock()

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
YELLOW = (240, 220, 50)
GREEN = (50, 200, 80)
GREY = (80, 80, 80)
DARKGREY = (40, 40, 40)


# ----------------------------------
# 신호 Phase 정의
# ----------------------------------
PHASE_SEQUENCE = [
    ("P1_MAIN_GREEN", 6),
    ("P1_YELLOW", 2),
    ("P2_MAIN_LEFT", 4),
    ("P2_YELLOW", 2),
    ("P3_SIDE_GREEN", 5),
    ("P3_YELLOW", 2),
]

TOTAL_PHASE_TIME = sum([d for _, d in PHASE_SEQUENCE])


# ----------------------------------
# 신호등 상태 계산
# ----------------------------------
def get_light_states(phase_name):
    state = {
        "M_LR_R": True,  "M_LR_Y": False, "M_LR_G": False, "M_LR_LEFT": False,
        "M_RL_R": True,  "M_RL_Y": False, "M_RL_G": False,
        "S_R": True,     "S_Y": False,    "S_G": False,
    }

    if phase_name == "P1_MAIN_GREEN":
        state.update({
            "M_LR_R": False, "M_LR_G": True,
            "M_RL_R": False, "M_RL_G": True,
        })

    elif phase_name == "P1_YELLOW":
        state.update({
            "M_LR_R": False, "M_LR_Y": True,
            "M_RL_R": False, "M_RL_Y": True,
        })

    elif phase_name == "P2_MAIN_LEFT":
        state.update({
            "M_LR_R": True, "M_LR_LEFT": True,
            "M_RL_R": True,
        })

    elif phase_name == "P2_YELLOW":
        state.update({
            "M_LR_R": True, "M_LR_Y": True,
            "M_RL_R": True,
        })

    elif phase_name == "P3_SIDE_GREEN":
        state.update({
            "S_R": False, "S_G": True
        })

    elif phase_name == "P3_YELLOW":
        state.update({
            "S_R": False, "S_Y": True
        })

    return state


# ----------------------------------
# 신호등 그리기 함수들
# ----------------------------------
def draw_signal_box_4light(x, y, title):
    pygame.draw.rect(SCREEN, BLACK, (x, y, 120, 220), border_radius=10)
    label = FONT.render(title, True, WHITE)
    SCREEN.blit(label, (x, y - 22))


def draw_signal_box_3light(x, y, title):
    pygame.draw.rect(SCREEN, BLACK, (x, y, 120, 170), border_radius=10)
    label = FONT.render(title, True, WHITE)
    SCREEN.blit(label, (x, y - 22))


def draw_light(x, y, color, on):
    radius = 16
    if on:
        pygame.draw.circle(SCREEN, color, (x, y), radius)
    else:
        pygame.draw.circle(SCREEN, GREY, (x, y), radius)
        pygame.draw.circle(SCREEN, color, (x, y), radius, 2)


def draw_arrow_light(x, y, on):
    radius = 16
    if on:
        pygame.draw.circle(SCREEN, GREEN, (x, y), radius)
    else:
        pygame.draw.circle(SCREEN, GREY, (x, y), radius)
        pygame.draw.circle(SCREEN, GREEN, (x, y), radius, 2)

    arrow_text = FONT.render("←", True, WHITE if on else GREEN)
    SCREEN.blit(arrow_text, arrow_text.get_rect(center=(x, y)))


def draw_horizontal_signal(x, y, title, state):
    pygame.draw.rect(SCREEN, BLACK, (x, y, 180, 80), border_radius=10)
    SCREEN.blit(FONT.render(title, True, WHITE), (x, y - 22))

    positions = [
        (x + 40, y + 40, RED, state["S_R"]),
        (x + 90, y + 40, YELLOW, state["S_Y"]),
        (x + 140, y + 40, GREEN, state["S_G"]),
    ]

    for px, py, color, on in positions:
        draw_light(px, py, color, on)


# ----------------------------------
# 교차로(삼거리) 그리기
# ----------------------------------
def draw_intersection(base_x, base_y, name, state):
    # 메인 왼→오른
    x1, y1 = base_x, base_y
    draw_signal_box_4light(x1, y1, f"{name} L→R")
    draw_light(x1+60, y1+40, RED, state["M_LR_R"])
    draw_light(x1+60, y1+85, YELLOW, state["M_LR_Y"])
    draw_light(x1+60, y1+130, GREEN, state["M_LR_G"])
    draw_arrow_light(x1+60, y1+175, state["M_LR_LEFT"])

    # 메인 오른→왼
    x2, y2 = base_x + 150, base_y + 40
    draw_signal_box_3light(x2, y2, f"{name} R→L")
    draw_light(x2+60, y2+35, RED, state["M_RL_R"])
    draw_light(x2+60, y2+80, YELLOW, state["M_RL_Y"])
    draw_light(x2+60, y2+125, GREEN, state["M_RL_G"])

    # 지선
    draw_horizontal_signal(base_x + 30, base_y - 120, f"{name} Side", state)


# ----------------------------------
# Phase 계산 (Offset 포함)
# ----------------------------------
def get_phase_from_time(t):
    t = t % TOTAL_PHASE_TIME

    cum = 0
    for name, duration in PHASE_SEQUENCE:
        if cum <= t < cum + duration:
            return name, (cum + duration - t)
        cum += duration

    return PHASE_SEQUENCE[-1][0], 0


# ----------------------------------
# 메인 루프
# ----------------------------------
def main():
    start_time = time.time()

    # 각 교차로 offset (초)
    offsets = {
        "A": 0,
        "B": 4,
        "C": 8,
    }

    while True:
        CLOCK.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        now = time.time() - start_time

        SCREEN.fill((20, 20, 20))

        # 교차로 A (상단 중앙)
        phaseA, remainA = get_phase_from_time(now + offsets["A"])
        draw_intersection(450, 300, "교차로1", get_light_states(phaseA))

        # 교차로 B (좌측 아래)
        phaseB, remainB = get_phase_from_time(now + offsets["B"])
        draw_intersection(50, 650, "교차로2", get_light_states(phaseB))

        # 교차로 C (우측 아래)
        phaseC, remainC = get_phase_from_time(now + offsets["C"])
        draw_intersection(900, 650, "교차로3", get_light_states(phaseC))

        pygame.display.flip()


if __name__ == "__main__":
    main()
