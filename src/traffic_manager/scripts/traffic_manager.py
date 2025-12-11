import pygame
import sys
import time


# ----------------------------------
# Signal state calculation helpers
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
    def __init__(self, width: int = 1240, height: int = 1020) -> None:
        # Screen setup
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3-Way T-Intersection Traffic Simulator")

        self.font = pygame.font.SysFont("malgungothic", 18)
        self.clock = pygame.time.Clock()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (220, 50, 50)
        self.YELLOW = (240, 220, 50)
        self.GREEN = (50, 200, 80)
        self.GREY = (80, 80, 80)
        self.DARKGREY = (40, 40, 40)

        # Phase sequence definition
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
    # Signal drawing helpers
    # ----------------------------------
    def _draw_signal_box_4light(self, surface, x, y, title, show_label=True):
        pygame.draw.rect(surface, self.BLACK, (x, y, 120, 220), border_radius=10)
        if show_label and title:
            label = self.font.render(title, True, self.WHITE)
            surface.blit(label, (x, y - 22))

    def _draw_signal_box_3light(self, surface, x, y, title, show_label=True):
        pygame.draw.rect(surface, self.BLACK, (x, y, 120, 170), border_radius=10)
        if show_label and title:
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

        arrow_text = self.font.render("<-", True, self.WHITE if on else self.GREEN)
        surface.blit(arrow_text, arrow_text.get_rect(center=(x, y)))

    def _draw_horizontal_signal(self, surface, x, y, title, state, show_label=True):
        pygame.draw.rect(surface, self.BLACK, (x, y, 180, 80), border_radius=10)
        if show_label and title:
            surface.blit(self.font.render(title, True, self.WHITE), (x, y - 22))

        positions = [
            (x + 40, y + 40, self.RED, state["S_R"]),
            (x + 90, y + 40, self.YELLOW, state["S_Y"]),
            (x + 140, y + 40, self.GREEN, state["S_G"]),
        ]

        for px, py, color, on in positions:
            self._draw_light(surface, px, py, color, on)

    def _draw_intersection_raw(self, surface, base_x, base_y, name, state, show_labels=True):
        # Main left-to-right
        x1, y1 = base_x, base_y
        self._draw_signal_box_4light(surface, x1, y1, f"{name} L->R", show_label=show_labels)
        self._draw_light(surface, x1 + 60, y1 + 40, self.RED, state["M_LR_R"])
        self._draw_light(surface, x1 + 60, y1 + 85, self.YELLOW, state["M_LR_Y"])
        self._draw_light(surface, x1 + 60, y1 + 130, self.GREEN, state["M_LR_G"])
        self._draw_arrow_light(surface, x1 + 60, y1 + 175, state["M_LR_LEFT"])

        # Main right-to-left
        x2, y2 = base_x + 150, base_y + 40
        self._draw_signal_box_3light(surface, x2, y2, f"{name} R->L", show_label=show_labels)
        self._draw_light(surface, x2 + 60, y2 + 35, self.RED, state["M_RL_R"])
        self._draw_light(surface, x2 + 60, y2 + 80, self.YELLOW, state["M_RL_Y"])
        self._draw_light(surface, x2 + 60, y2 + 125, self.GREEN, state["M_RL_G"])

        # Side road
        self._draw_horizontal_signal(surface, base_x + 30, base_y - 120, f"{name} Side", state, show_label=show_labels)

    # ----------------------------------
    # Intersection drawing with optional rotation
    # ----------------------------------
    def draw_intersection(self, base_x, base_y, name, state, angle: int = 0):
        """
        base_x/base_y correspond to the reference corner of the left-to-right signal box.
        angle is in degrees (pygame convention: positive is CCW).
        """
        show_labels = angle == 0
        if angle == 0:
            # Draw directly on the main screen
            self._draw_intersection_raw(self.screen, base_x, base_y, name, state, show_labels=show_labels)
            anchor_x = base_x + 150
            anchor_y = base_y - 10
        else:
            # Draw on a temporary surface and rotate/blit
            # (use a generous canvas so rotation does not clip)
            temp_width, temp_height = 400, 400
            temp_surface = pygame.Surface((temp_width, temp_height), pygame.SRCALPHA)

            # Internal coordinates within the temporary surface
            local_base_x, local_base_y = 80, 140
            self._draw_intersection_raw(temp_surface, local_base_x, local_base_y, name, state, show_labels=False)

            rotated = pygame.transform.rotate(temp_surface, angle)

            # Approximate alignment with the requested base point
            center_x = base_x + 150
            center_y = base_y + 120

            rect = rotated.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated, rect.topleft)
            anchor_x = rect.centerx
            anchor_y = rect.top - 10

        label_surface = self.font.render(name, True, self.WHITE)
        label_rect = label_surface.get_rect(center=(anchor_x, anchor_y))
        self.screen.blit(label_surface, label_rect)

    # ----------------------------------
    # Phase computation (per-intersection offset)
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
    # Main loop
    # ----------------------------------
    def run(self):
        start_time = time.time()

        # Intersection offsets (seconds)
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

            # Intersection A (top center) rotated 180 degrees
            phaseA, remainA = self.get_phase_from_time(now + offsets["A"])
            self.draw_intersection(450, 100, "Intersection 1", get_light_states(phaseA), angle=180)

            # Intersection B (bottom left) rotated -90 degrees
            phaseB, remainB = self.get_phase_from_time(now + offsets["B"])
            self.draw_intersection(70, 570, "Intersection 2", get_light_states(phaseB), angle=-90)

            # Intersection C (bottom right) rotated +90 degrees
            phaseC, remainC = self.get_phase_from_time(now + offsets["C"])
            self.draw_intersection(880, 570, "Intersection 3", get_light_states(phaseC), angle=90)

            pygame.display.flip()


def main():
    pygame.init()
    simulator = TrafficSimulator()
    simulator.run()


if __name__ == "__main__":
    main()
