from laser import SweepingLaser


class Level:
    def __init__(self, name, start_x, goal_x, lasers=None):
        self.name = name
        self.start_x = start_x
        self.goal_x = goal_x
        self.lasers = lasers or []

    def reset_player(self, player):
        player.reset_position(self.start_x)

    def update(self, dt):
        for laser in self.lasers:
            laser.update(dt)

    def completed(self, player):
        if self.goal_x > self.start_x:
            return player.x >= self.goal_x

        return player.x <= self.goal_x


def make_levels():
    return [
        Level(
            name="Move Right",
            start_x=80,
            goal_x=920,
            lasers=[],
        ),
        Level(
            name="Move Left",
            start_x=920,
            goal_x=80,
            lasers=[],
        ),
        Level(
            name="Timing Right",
            start_x=80,
            goal_x=920,
            lasers=[
                SweepingLaser(x=500, wait_time=1.2),
            ],
        ),
        Level(
            name="Timing Left",
            start_x=920,
            goal_x=80,
            lasers=[
                SweepingLaser(x=500, wait_time=1.2),
            ],
        ),
    ]