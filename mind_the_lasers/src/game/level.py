from mind_the_lasers.src.game.laser import SweepingLaser


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

        # Tutorial
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

        # Single laser
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

        # Two synchronized lasers
        Level(
            name="Double Timing Right",
            start_x=80,
            goal_x=920,
            lasers=[
                SweepingLaser(x=350, wait_time=1.5),
                SweepingLaser(x=700, wait_time=1.5),
            ],
        ),

        Level(
            name="Double Timing Left",
            start_x=920,
            goal_x=80,
            lasers=[
                SweepingLaser(x=650, wait_time=1.5),
                SweepingLaser(x=300, wait_time=1.5),
            ],
        ),

        # Two alternating lasers
        Level(
            name="Alternating Right",
            start_x=80,
            goal_x=920,
            lasers=[
                SweepingLaser(x=350, wait_time=2.0, initial_delay=0.0),
                SweepingLaser(x=700, wait_time=2.0, initial_delay=1.0),
            ],
        ),

        Level(
            name="Alternating Left",
            start_x=920,
            goal_x=80,
            lasers=[
                SweepingLaser(x=650, wait_time=2.0, initial_delay=0.0),
                SweepingLaser(x=300, wait_time=2.0, initial_delay=1.0),
            ],
        ),

        # Three alternating lasers
        Level(
            name="Triple Timing Right",
            start_x=80,
            goal_x=920,
            lasers=[
                SweepingLaser(x=250, wait_time=2.4, initial_delay=0.0),
                SweepingLaser(x=500, wait_time=2.4, initial_delay=0.8),
                SweepingLaser(x=750, wait_time=2.4, initial_delay=1.6),
            ],
        ),

        Level(
            name="Triple Timing Left",
            start_x=920,
            goal_x=80,
            lasers=[
                SweepingLaser(x=750, wait_time=2.4, initial_delay=0.0),
                SweepingLaser(x=500, wait_time=2.4, initial_delay=0.8),
                SweepingLaser(x=250, wait_time=2.4, initial_delay=1.6),
            ],
        ),

        # Final challenge
        Level(
            name="Mind the Lasers",
            start_x=80,
            goal_x=920,
            lasers=[
                SweepingLaser(x=180, wait_time=1.6, initial_delay=0.0, grow_speed=700),
                SweepingLaser(x=340, wait_time=1.6, initial_delay=0.4, grow_speed=700),
                SweepingLaser(x=500, wait_time=1.6, initial_delay=0.8, grow_speed=700),
                SweepingLaser(x=660, wait_time=1.6, initial_delay=1.2, grow_speed=700),
                SweepingLaser(x=820, wait_time=1.6, initial_delay=1.6, grow_speed=700),
            ],
        ),
    ]