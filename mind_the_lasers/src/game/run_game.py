import argparse

import pygame

from mind_the_lasers.src.game.settings import *
from mind_the_lasers.src.game.game import Game
from mind_the_lasers.src.game.mode_select import ModeSelect, ErrPModeSelect
from mind_the_lasers.src.game.training_mode import TrainingMode
from mind_the_lasers.src.game.errp_mode import ErrPMode
from mind_the_lasers.src.game.input_controller import ContinuousKeyboardController, UDPController

from mind_the_lasers.src.game.laser import SweepingLaser

from mind_the_lasers.src.stream.marker_sender import create_marker_sender


def present_scaled_frame(logical_screen):
    """Fit the fixed 1000x600 game canvas inside the current display window.

    All game geometry remains in logical coordinates. Preserve its aspect ratio
    so resizing/fullscreen cannot distort the player, targets, or UI.
    """
    window = pygame.display.get_surface()
    window_width, window_height = window.get_size()
    if window_width <= 0 or window_height <= 0:
        return  # Minimized window; nothing to present.

    scale = min(window_width / WIDTH, window_height / HEIGHT)
    scaled_width = min(window_width, max(1, round(WIDTH * scale)))
    scaled_height = min(window_height, max(1, round(HEIGHT * scale)))

    if (scaled_width, scaled_height) == (WIDTH, HEIGHT):
        frame = logical_screen
    else:
        frame = pygame.transform.smoothscale(
            logical_screen, (scaled_width, scaled_height)
        )

    window.fill(BG_COLOR)  # Letterbox/pillarbox when aspect ratios differ.
    window.blit(
        frame,
        ((window_width - scaled_width) // 2,
         (window_height - scaled_height) // 2),
    )
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Run Mind the Lasers")
    parser.add_argument(
        "--controller",
        choices=["pipeline", "keyboard", "local_keyboard", "both"],
        default="pipeline",
        help="Input source for the game.",
    )
    args = parser.parse_args()

    pygame.init()

    windowed_size = (WIDTH, HEIGHT)
    fullscreen = False
    pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
    # Game modes always draw onto this fixed-size canvas, not the OS window.
    # This keeps positions, speeds, and ErrP thresholds resolution-independent.
    screen = pygame.Surface((WIDTH, HEIGHT)).convert()

    pygame.display.set_caption(
        "Mind the Lasers"
    )

    clock = pygame.time.Clock()

    controller = UDPController()

    marker_sender = create_marker_sender(
        transport=MARKER_TRANSPORT,
        udp_ip=MARKER_UDP_IP,
        udp_port=MARKER_UDP_PORT,
    )

    mode_select = ModeSelect(screen)
    errp_mode_select = ErrPModeSelect(screen)

    menu_lasers = [
        SweepingLaser(x=90,          wait_time=1.0, initial_delay=0.0, grow_speed=800, beam_width=18, active_time=0.3),
        SweepingLaser(x=180,         wait_time=1.0, initial_delay=0.5, grow_speed=800, beam_width=18, active_time=0.3),
        SweepingLaser(x=WIDTH - 180, wait_time=1.0, initial_delay=1.0, grow_speed=800, beam_width=18, active_time=0.3),
        SweepingLaser(x=WIDTH - 90,  wait_time=1.0, initial_delay=1.5, grow_speed=800, beam_width=18, active_time=0.3),
    ]

    current_mode = "menu"

    game = None
    training_mode = None
    errp_mode = None

    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Pygame 2 resizes the display surface automatically; don't resize
            # the logical canvas or reconstruct any active mode on this event.
            if event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
                if not fullscreen:
                    windowed_size = pygame.display.get_window_size()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    if fullscreen:
                        pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
                        fullscreen = False
                    else:
                        windowed_size = pygame.display.get_window_size()
                        # (0, 0) requests the current desktop resolution.
                        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        fullscreen = True
                    continue  # F11 is a display shortcut, not game input.

                if event.key == pygame.K_ESCAPE:
                    if current_mode == "menu":
                        running = False

                    elif current_mode == "errp_select":
                        current_mode = "menu"

                    else:
                        if training_mode is not None:
                            training_mode.close()

                        if errp_mode is not None:
                            errp_mode.close()

                        current_mode = "menu"

                        game = None
                        training_mode = None
                        errp_mode = None

                if (
                    current_mode == "training"
                    and training_mode is not None
                    and training_mode.is_complete()
                    and event.key == pygame.K_RETURN
                ):
                    training_mode.close()
                    training_mode = None
                    current_mode = "menu"

                if (
                    current_mode == "errp"
                    and errp_mode is not None
                    and errp_mode.is_complete()
                    and event.key == pygame.K_RETURN
                ):
                    errp_mode.close()
                    errp_mode = None
                    current_mode = "menu"

                if current_mode == "errp" and errp_mode is not None:
                    errp_mode.handle_event(event)

                if (
                    current_mode == "play"
                    and game is not None
                    and game.game_over
                    and event.key == pygame.K_RETURN
                ):
                    game = None
                    controller.reset()
                    current_mode = "menu"

            if current_mode == "menu":
                selected_mode = mode_select.handle_event(event)

                if selected_mode == "training":
                    controller.reset()
                    training_mode = TrainingMode(
                        screen,
                        controller,
                        marker_sender
                    )

                    current_mode = "training"

                elif selected_mode == "play":
                    controller.reset()

                    play_controller = (
                        ContinuousKeyboardController()
                        if args.controller == "local_keyboard"
                        else controller
                    )

                    game = Game(
                        screen,
                        play_controller,
                    )

                    current_mode = "play"

                elif selected_mode == "errp":
                    current_mode = "errp_select"

            elif current_mode == "errp_select":
                sub_mode = errp_mode_select.handle_event(event)

                if sub_mode is not None:
                    controller.reset()

                    errp_controller = (
                        ContinuousKeyboardController()
                        if args.controller == "local_keyboard"
                        else controller
                    )

                    errp_mode = ErrPMode(
                        screen,
                        errp_controller,
                        marker_sender,
                        sub_mode=sub_mode,
                    )

                    current_mode = "errp"
            

        if current_mode == "menu":
            for laser in menu_lasers:
                laser.update(dt)

            mode_select.draw()

            for laser in menu_lasers:
                laser.draw(screen)

        elif current_mode == "errp_select":
            errp_mode_select.draw()

        elif current_mode == "training":
            training_mode.update(dt)
            training_mode.draw()

        elif current_mode == "play":
            if game.game_over:
                game.draw_summary()
            else:
                game.update(dt)
                game.draw()

        elif current_mode == "errp":
            errp_mode.update(dt)
            errp_mode.draw()

        present_scaled_frame(screen)

    if training_mode is not None:
        training_mode.close()

    if errp_mode is not None:
        errp_mode.close()

    marker_sender.close()
    pygame.quit()


if __name__ == "__main__":
    main()