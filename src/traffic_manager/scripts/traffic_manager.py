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
            ("P1_MAIN_GREEN", 25),
            ("P1_YELLOW", 3),
            ("P2_MAIN_LEFT", 10),
            ("P2_YELLOW", 3),
            ("P3_SIDE_GREEN", 15),
            ("P3_YELLOW", 3),
        ]
        self.total_phase_time = sum(d for _, d in self.phase_sequence)

    # ----------------------------------
    # 신호등 그리기 메서드들 (임의 Surface 지원)
    # ----------------------------------
    def _draw_signal_box_4light(self, surface, x, y, title):
        pygame.draw.rect(surface, self.BLACK, (x, y, 120, 220), border_radius=10)
        label = self.font.render(title, True, self.WHITE)
        surface.blit(label, (x, y - 22))

    def _draw_signal_box_3light(self, surface, x, y, title):
        pygame.draw.rect(surface, self.BLACK, (x, y, 120, 170), border_radius=10)
        label = self.font.render(title, True, self.WHITE)
        surface.blit(label, (x, y - 22))

    def _draw_light(self, surface, x, y, color, on):
        radius = 16
        if on:
            pygame.draw.circle(surface, color, (x, y), radius)
        else:
            pygame.draw.circle(surface, self.GREY, (x, y), radius)
            pygame.draw.circle(surface, color, (x, y), radius, 2)

    def _draw_arrow_light(self, surface, x, y, on):
        radius = 16
        if on:
            pygame.draw.circle(surface, self.GREEN, (x, y), radius)
        else:
            pygame.draw.circle(surface, self.GREY, (x, y), radius)
            pygame.draw.circle(surface, self.GREEN, (x, y), radius, 2)

        arrow_text = self.font.render("←", True, self.WHITE if on else self.GREEN)
        surface.blit(arrow_text, arrow_text.get_rect(center=(x, y)))

    def _draw_horizontal_signal(self, surface, x, y, title, state):
        pygame.draw.rect(surface, self.BLACK, (x, y, 180, 80), border_radius=10)
        surface.blit(self.font.render(title, True, self.WHITE), (x, y - 22))

        positions = [
            (x + 40, y + 40, self.RED, state["S_R"]),
            (x + 90, y + 40, self.YELLOW, state["S_Y"]),
            (x + 140, y + 40, self.GREEN, state["S_G"]),
        ]

        for px, py, color, on in positions:
            self._draw_light(surface, px, py, color, on)

    def _draw_intersection_raw(self, surface, base_x, base_y, name, state):
        # 메인 왼→오른
        x1, y1 = base_x, base_y
        self._draw_signal_box_4light(surface, x1, y1, f"{name} L→R")
        self._draw_light(surface, x1 + 60, y1 + 40, self.RED, state["M_LR_R"])
        self._draw_light(surface, x1 + 60, y1 + 85, self.YELLOW, state["M_LR_Y"])
        self._draw_light(surface, x1 + 60, y1 + 130, self.GREEN, state["M_LR_G"])
        self._draw_arrow_light(surface, x1 + 60, y1 + 175, state["M_LR_LEFT"])

        # 메인 오른→왼
        x2, y2 = base_x + 150, base_y + 40
        self._draw_signal_box_3light(surface, x2, y2, f"{name} R→L")
        self._draw_light(surface, x2 + 60, y2 + 35, self.RED, state["M_RL_R"])
        self._draw_light(surface, x2 + 60, y2 + 80, self.YELLOW, state["M_RL_Y"])
        self._draw_light(surface, x2 + 60, y2 + 125, self.GREEN, state["M_RL_G"])

        # 지선
        self._draw_horizontal_signal(surface, base_x + 30, base_y - 120, f"{name} Side", state)

    # ----------------------------------
    # 교차로(삼거리) 그리기 (회전 지원)
    # ----------------------------------
    def draw_intersection(self, base_x, base_y, name, state, angle: int = 0):
        """
        base_x, base_y는 '메인 L→R 신호 상자의 왼쪽 위 좌표' 기준.
        angle은 도 단위이며, pygame의 규칙을 따라 양수는 반시계 방향 회전.
        """
        if angle == 0:
            # 기존과 동일하게 바로 화면에 그림
            self._draw_intersection_raw(self.screen, base_x, base_y, name, state)
        else:
            # 임시 Surface 위에 그리고 전체를 회전시켜서 blit
            # (회전 시 잘리지 않도록 넉넉한 크기로 확보)
            temp_width, temp_height = 400, 400
            temp_surface = pygame.Surface((temp_width, temp_height), pygame.SRCALPHA)

            # temp_surface 내부에서의 교차로 위치 (적당한 마진 포함)
            # 필요하면 이 값을 조절해서 안쪽 배치를 미세 조정할 수 있음
            local_base_x, local_base_y = 80, 140
            self._draw_intersection_raw(temp_surface, local_base_x, local_base_y, name, state)

            rotated = pygame.transform.rotate(temp_surface, angle)

            # 원래 base_x, base_y 기준으로 교차로 중심을 대략 맞춤
            center_x = base_x + 150  # 대략 교차로 중심 x
            center_y = base_y + 120  # 대략 교차로 중심 y

            rect = rotated.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated, rect.topleft)

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
            "A": 20,
            "B": 0,
            "C": 38,
        }

        running = True
        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time() - start_time

            self.screen.fill((20, 20, 20))

            # 교차로 A (상단 중앙) - 180도 회전
            phaseA, remainA = self.get_phase_from_time(now + offsets["A"])
            self.draw_intersection(450, 100, "교차로1", get_light_states(phaseA), angle=180)

            # 교차로 B (좌측 아래) - 시계 방향 90도 회전
            phaseB, remainB = self.get_phase_from_time(now + offsets["B"])
            self.draw_intersection(70, 570, "교차로2", get_light_states(phaseB), angle=-90)

            # 교차로 C (우측 아래) - 반시계 방향 90도 회전
            phaseC, remainC = self.get_phase_from_time(now + offsets["C"])
            self.draw_intersection(880, 570, "교차로3", get_light_states(phaseC), angle=90)

            pygame.display.flip()


def main():
    pygame.init()
    simulator = TrafficSimulator()
    simulator.run()


if __name__ == "__main__":
    main()
