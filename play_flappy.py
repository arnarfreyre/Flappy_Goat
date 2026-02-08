"""
Standalone Flappy Bird player with Pygame GUI.
State machine: MENU → READY → PLAYING → GAME_OVER → READY (loop)
"""

import sys
import os

# Add PLE to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, "itml-project2"))

import pygame
from ple.games.flappybird import FlappyBird
from ple import PLE

# --- Constants ---
WIDTH, HEIGHT = 288, 512
FPS = 30

# Colors
SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (76, 175, 80)
GREEN_HOVER = (102, 195, 105)
RED = (211, 47, 47)
RED_HOVER = (229, 85, 85)
OVERLAY = (0, 0, 0, 150)
YELLOW = (255, 193, 7)

# States
MENU = "menu"
READY = "ready"
PLAYING = "playing"
GAME_OVER = "game_over"


class Button:
    def __init__(self, x, y, w, h, text, color, hover_color, text_color=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color

    def draw(self, surface, font):
        mx, my = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mx, my)
        color = self.hover_color if hovered else self.color

        # Rounded rectangle
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, WHITE, self.rect, width=2, border_radius=8)

        # Text centered
        txt = font.render(self.text, True, self.text_color)
        tx = self.rect.centerx - txt.get_width() // 2
        ty = self.rect.centery - txt.get_height() // 2
        surface.blit(txt, (tx, ty))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


def draw_text_centered(surface, text, font, color, y):
    txt = font.render(text, True, color)
    x = (WIDTH - txt.get_width()) // 2
    surface.blit(txt, (x, y))


def draw_overlay(surface):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill(OVERLAY)
    surface.blit(overlay, (0, 0))


def main():
    pygame.init()

    # Create game and PLE with display enabled
    game = FlappyBird(width=WIDTH, height=HEIGHT)
    p = PLE(game, fps=FPS, display_screen=True, force_fps=False)
    p.init()

    action_set = p.getActionSet()
    flap_action = action_set[0]
    noop_action = action_set[1]

    # Get the display surface PLE created
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Flappy Bird")

    # Fonts
    font_large = pygame.font.SysFont("Arial", 48, bold=True)
    font_medium = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 20)

    # Buttons
    btn_w, btn_h = 160, 50
    cx = WIDTH // 2 - btn_w // 2

    play_btn = Button(cx, 320, btn_w, btn_h, "Play", GREEN, GREEN_HOVER)
    start_btn = Button(cx, 350, btn_w, btn_h, "Start", GREEN, GREEN_HOVER)
    again_btn = Button(cx - 90, 370, btn_w, btn_h, "Play Again", GREEN, GREEN_HOVER)
    quit_btn = Button(cx + 90, 370, btn_w, btn_h, "Quit", RED, RED_HOVER)

    clock = pygame.time.Clock()
    state = MENU
    score = 0

    running = True
    while running:
        # --- Event handling ---
        click_pos = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click_pos = event.pos

        if not running:
            break

        # --- State machine ---
        if state == MENU:
            screen.fill(SKY_BLUE)
            draw_text_centered(screen, "Flappy Bird", font_large, YELLOW, 120)
            draw_text_centered(screen, "Flappy Bird", font_large, BLACK, 123)
            draw_text_centered(screen, "Flappy Bird", font_large, YELLOW, 120)
            play_btn.draw(screen, font_medium)

            if click_pos and play_btn.is_clicked(click_pos):
                p.reset_game()
                state = READY

        elif state == READY:
            # Render game's initial frame
            p.act(noop_action)
            draw_overlay(screen)
            draw_text_centered(screen, "Get Ready!", font_large, YELLOW, 100)
            draw_text_centered(screen, "SPACE / UP to flap", font_small, WHITE, 260)
            start_btn.draw(screen, font_medium)

            if click_pos and start_btn.is_clicked(click_pos):
                state = PLAYING

        elif state == PLAYING:
            # Read keys for flap input
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                action = flap_action
            else:
                action = noop_action

            p.act(action)
            score = p.score()

            # Draw score on top of game
            score_txt = font_medium.render(str(int(score)), True, WHITE)
            score_shadow = font_medium.render(str(int(score)), True, BLACK)
            sx = WIDTH // 2 - score_txt.get_width() // 2
            screen.blit(score_shadow, (sx + 2, 22))
            screen.blit(score_txt, (sx, 20))

            if p.game_over():
                state = GAME_OVER

        elif state == GAME_OVER:
            draw_overlay(screen)
            draw_text_centered(screen, "Game Over", font_large, RED, 100)
            draw_text_centered(screen, f"Score: {int(score)}", font_medium, YELLOW, 200)
            again_btn.draw(screen, font_medium)
            quit_btn.draw(screen, font_medium)

            if click_pos:
                if again_btn.is_clicked(click_pos):
                    p.reset_game()
                    score = 0
                    state = READY
                elif quit_btn.is_clicked(click_pos):
                    running = False

        pygame.display.update()
        clock.tick(FPS)

    # Cleanup
    pygame.display.quit()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
