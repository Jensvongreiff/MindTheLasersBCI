import pygame

from mind_the_lasers.src.game.settings import *
from mind_the_lasers.src.game.game import Game
from mind_the_lasers.src.game.mode_select import ModeSelect
from mind_the_lasers.src.game.training_mode import TrainingMode
from mind_the_lasers.src.game.input_controller import UDPController

from mind_the_lasers.src.game.laser import SweepingLaser

from mind_the_lasers.src.stream.marker_sender import LSLMarkerSender


def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "Mind the Lasers"
    )

    clock = pygame.time.Clock()

    controller = UDPController()

    marker_sender = LSLMarkerSender()

    mode_select = ModeSelect(screen)

    menu_lasers = [
        SweepingLaser(x=90,          wait_time=1.0, initial_delay=0.0, grow_speed=800, beam_width=18, active_time=0.3),
        SweepingLaser(x=180,         wait_time=1.0, initial_delay=0.5, grow_speed=800, beam_width=18, active_time=0.3),
        SweepingLaser(x=WIDTH - 180, wait_time=1.0, initial_delay=1.0, grow_speed=800, beam_width=18, active_time=0.3),
        SweepingLaser(x=WIDTH - 90,  wait_time=1.0, initial_delay=1.5, grow_speed=800, beam_width=18, active_time=0.3),
    ]

    current_mode = "menu"

    game = None
    training_mode = None

    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if current_mode == "menu":
                        running = False

                    else:
                        if training_mode is not None:
                            training_mode.close()

                        current_mode = "menu"

                        game = None
                        training_mode = None

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
                        marker_sender,
                        trials_per_command=5,
                    )

                    current_mode = "training"

                elif selected_mode == "play":
                    controller.reset()

                    game = Game(
                        screen,
                        controller,
                    )

                    current_mode = "play"
            

        if current_mode == "menu":
            for laser in menu_lasers:
                laser.update(dt)

            mode_select.draw()

            for laser in menu_lasers:
                laser.draw(screen)

        elif current_mode == "training":
            training_mode.update(dt)
            training_mode.draw()

        elif current_mode == "play":
            if game.game_over:
                game.draw_summary()
            else:
                game.update(dt)
                game.draw()

        pygame.display.flip()

    if training_mode is not None:
        training_mode.close()

    pygame.quit()


if __name__ == "__main__":
    main()