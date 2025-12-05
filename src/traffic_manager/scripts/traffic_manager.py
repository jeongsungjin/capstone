import pygame
import sys
import time


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


class TrafficSimulator:
    def __init__(self, width: int = 1240, height: int = 900) -> None:
        # 기본 설정
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3-Way T-Intersection Traffic Simulator")

        self.font = pygame.font.SysFont("malgungothic", 18)
        self.clock = pygame.time.Clock()

        # 색상 정의
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (220, 50, 50)
        self.YELLOW = (240, 220, 50)
        self.GREEN = (50, 200, 80)
        self.GREY = (80, 80, 80)
        self.DARKGREY = (40, 40, 40)

        # 신호 Phase 정의
        self.phase_sequence = [
            ("P1_MAIN_GREEN", 6),
            ("P1_YELLOW", 2),
            ("P2_MAIN_LEFT", 4),
            ("P2_YELLOW", 2),
            ("P3_SIDE_GREEN", 5),
            ("P3_YELLOW", 2),
        ]
        self.total_phase_time = sum(d for _, d in self.phase_sequence)

    # ----------------------------------
    # 신호등 그리기 메서드들
    # ----------------------------------
    def draw_signal_box_4light(self, x, y, title):
        pygame.draw.rect(self.screen, self.BLACK, (x, y, 120, 220), border_radius=10)
        label = self.font.render(title, True, self.WHITE)
        self.screen.blit(label, (x, y - 22))

    def draw_signal_box_3light(self, x, y, title):
        pygame.draw.rect(self.screen, self.BLACK, (x, y, 120, 170), border_radius=10)
        label = self.font.render(title, True, self.WHITE)
        self.screen.blit(label, (x, y - 22))

    def draw_light(self, x, y, color, on):
        radius = 16
        if on:
            pygame.draw.circle(self.screen, color, (x, y), radius)
        else:
            pygame.draw.circle(self.screen, self.GREY, (x, y), radius)
            pygame.draw.circle(self.screen, color, (x, y), radius, 2)

    def draw_arrow_light(self, x, y, on):
        radius = 16
        if on:
            pygame.draw.circle(self.screen, self.GREEN, (x, y), radius)
        else:
            pygame.draw.circle(self.screen, self.GREY, (x, y), radius)
            pygame.draw.circle(self.screen, self.GREEN, (x, y), radius, 2)

        arrow_text = self.font.render("←", True, self.WHITE if on else self.GREEN)
        self.screen.blit(arrow_text, arrow_text.get_rect(center=(x, y)))

    def draw_horizontal_signal(self, x, y, title, state):
        pygame.draw.rect(self.screen, self.BLACK, (x, y, 180, 80), border_radius=10)
        self.screen.blit(self.font.render(title, True, self.WHITE), (x, y - 22))

        positions = [
            (x + 40, y + 40, self.RED, state["S_R"]),
            (x + 90, y + 40, self.YELLOW, state["S_Y"]),
            (x + 140, y + 40, self.GREEN, state["S_G"]),
        ]

        for px, py, color, on in positions:
            self.draw_light(px, py, color, on)

    # ----------------------------------
    # 교차로(삼거리) 그리기
    # ----------------------------------
    def draw_intersection(self, base_x, base_y, name, state):
        # 메인 왼→오른
        x1, y1 = base_x, base_y
        self.draw_signal_box_4light(x1, y1, f"{name} L→R")
        self.draw_light(x1 + 60, y1 + 40, self.RED, state["M_LR_R"])
        self.draw_light(x1 + 60, y1 + 85, self.YELLOW, state["M_LR_Y"])
        self.draw_light(x1 + 60, y1 + 130, self.GREEN, state["M_LR_G"])
        self.draw_arrow_light(x1 + 60, y1 + 175, state["M_LR_LEFT"])

        # 메인 오른→왼
        x2, y2 = base_x + 150, base_y + 40
        self.draw_signal_box_3light(x2, y2, f"{name} R→L")
        self.draw_light(x2 + 60, y2 + 35, self.RED, state["M_RL_R"])
        self.draw_light(x2 + 60, y2 + 80, self.YELLOW, state["M_RL_Y"])
        self.draw_light(x2 + 60, y2 + 125, self.GREEN, state["M_RL_G"])

        # 지선
        self.draw_horizontal_signal(base_x + 30, base_y - 120, f"{name} Side", state)

    # ----------------------------------
    # Phase 계산 (Offset 포함)
    # ----------------------------------
    def get_phase_from_time(self, t):
        t = t % self.total_phase_time

        cum = 0
        for name, duration in self.phase_sequence:
            if cum <= t < cum + duration:
                return name, (cum + duration - t)
            cum += duration

        return self.phase_sequence[-1][0], 0

    # ----------------------------------
    # 메인 루프
    # ----------------------------------
    def run(self):
        start_time = time.time()

        # 각 교차로 offset (초)
        offsets = {
            "A": 0,
            "B": 4,
            "C": 8,
        }

        running = True
        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time() - start_time

            self.screen.fill((20, 20, 20))

            # 교차로 A (상단 중앙)
            phaseA, remainA = self.get_phase_from_time(now + offsets["A"])
            self.draw_intersection(450, 300, "교차로1", get_light_states(phaseA))

            # 교차로 B (좌측 아래)
            phaseB, remainB = self.get_phase_from_time(now + offsets["B"])
            self.draw_intersection(50, 650, "교차로2", get_light_states(phaseB))

            # 교차로 C (우측 아래)
            phaseC, remainC = self.get_phase_from_time(now + offsets["C"])
            self.draw_intersection(900, 650, "교차로3", get_light_states(phaseC))

            pygame.display.flip()


def main():
    pygame.init()
    simulator = TrafficSimulator()
    simulator.run()


if __name__ == "__main__":
    main()
