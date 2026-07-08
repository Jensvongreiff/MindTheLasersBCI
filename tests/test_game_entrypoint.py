import unittest
from unittest.mock import patch

from mind_the_lasers.src.game import game as game_module


class GameEntrypointTests(unittest.TestCase):
    def test_main_starts_the_game(self):
        with patch.object(game_module, "Game") as mock_game_cls:
            game_module.main()

        mock_game_cls.assert_called_once_with()
        mock_game_cls.return_value.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
