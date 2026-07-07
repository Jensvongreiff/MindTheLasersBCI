import pygame

from settings import *
from game import Game
from mode_select import ModeSelect
from training_mode import TrainingMode
from input_controller import UDPController


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

    mode_select = ModeSelect(screen)

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
            mode_select.draw()

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