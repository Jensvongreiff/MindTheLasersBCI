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

        # 1-2 Tutorial
        Level("Move Right", 80, 920, []),
        Level("Move Left", 920, 80, []),

        # 3-4 Single laser
        Level("Timing Right", 80, 920, [
            SweepingLaser(500, wait_time=1.2)
        ]),
        Level("Timing Left", 920, 80, [
            SweepingLaser(500, wait_time=1.2)
        ]),

        # 5-6 Two synchronized
        Level("Double Right", 80, 920, [
            SweepingLaser(350, wait_time=1.5),
            SweepingLaser(700, wait_time=1.5),
        ]),
        Level("Double Left", 920, 80, [
            SweepingLaser(650, wait_time=1.5),
            SweepingLaser(300, wait_time=1.5),
        ]),

        # 7-8 Alternating
        Level("Alternating Right", 80, 920, [
            SweepingLaser(350, wait_time=2.0),
            SweepingLaser(700, wait_time=2.0, initial_delay=1.0),
        ]),
        Level("Alternating Left", 920, 80, [
            SweepingLaser(650, wait_time=2.0),
            SweepingLaser(300, wait_time=2.0, initial_delay=1.0),
        ]),

        # 9-10 Triple
        Level("Triple Right", 80, 920, [
            SweepingLaser(250, 2.4),
            SweepingLaser(500, 2.4, .8),
            SweepingLaser(750, 2.4, 1.6),
        ]),
        Level("Triple Left", 920, 80, [
            SweepingLaser(750, 2.4),
            SweepingLaser(500, 2.4, .8),
            SweepingLaser(250, 2.4, 1.6),
        ]),

        # 11-12 Fast center
        Level("Fast Right", 80, 920, [
            SweepingLaser(500, 1.0, grow_speed=700),
        ]),
        Level("Fast Left", 920, 80, [
            SweepingLaser(500, 1.0, grow_speed=700),
        ]),

        # 13-14 Fast double
        Level("Fast Double Right", 80, 920, [
            SweepingLaser(350, 1.2, grow_speed=700),
            SweepingLaser(700, 1.2, grow_speed=700),
        ]),
        Level("Fast Double Left", 920, 80, [
            SweepingLaser(650, 1.2, grow_speed=700),
            SweepingLaser(300, 1.2, grow_speed=700),
        ]),

        # 15-16 Four corridor
        Level("Corridor Right", 80, 920, [
            SweepingLaser(220, 2.4),
            SweepingLaser(420, 2.4, .6),
            SweepingLaser(620, 2.4, 1.2),
            SweepingLaser(820, 2.4, 1.8),
        ]),
        Level("Corridor Left", 920, 80, [
            SweepingLaser(780, 2.4),
            SweepingLaser(580, 2.4, .6),
            SweepingLaser(380, 2.4, 1.2),
            SweepingLaser(180, 2.4, 1.8),
        ]),

        # 17-18 Zigzag timing
        Level("Zigzag Right", 80, 920, [
            SweepingLaser(250, 1.5),
            SweepingLaser(500, 2.0, .5),
            SweepingLaser(750, 1.5, 1.0),
        ]),
        Level("Zigzag Left", 920, 80, [
            SweepingLaser(750, 1.5),
            SweepingLaser(500, 2.0, .5),
            SweepingLaser(250, 1.5, 1.0),
        ]),

        # 19-20 Dense
        Level("Dense Right", 80, 920, [
            SweepingLaser(180, 2.0),
            SweepingLaser(360, 2.0, .4),
            SweepingLaser(540, 2.0, .8),
            SweepingLaser(720, 2.0, 1.2),
            SweepingLaser(900, 2.0, 1.6),
        ]),
        Level("Dense Left", 920, 80, [
            SweepingLaser(900, 2.0),
            SweepingLaser(720, 2.0, .4),
            SweepingLaser(540, 2.0, .8),
            SweepingLaser(360, 2.0, 1.2),
            SweepingLaser(180, 2.0, 1.6),
        ]),

        # 21-22 Faster dense
        Level("Rapid Right", 80, 920, [
            SweepingLaser(200, 1.4, grow_speed=750),
            SweepingLaser(400, 1.4, .35, 750),
            SweepingLaser(600, 1.4, .70, 750),
            SweepingLaser(800, 1.4, 1.05, 750),
        ]),
        Level("Rapid Left", 920, 80, [
            SweepingLaser(800, 1.4, grow_speed=750),
            SweepingLaser(600, 1.4, .35, 750),
            SweepingLaser(400, 1.4, .70, 750),
            SweepingLaser(200, 1.4, 1.05, 750),
        ]),

        # 23-24 Offset chaos
        Level("Chaos Right", 80, 920, [
            SweepingLaser(220, 1.6, .0),
            SweepingLaser(420, 2.1, .4),
            SweepingLaser(620, 1.7, .9),
            SweepingLaser(820, 2.3, 1.2),
        ]),
        Level("Chaos Left", 920, 80, [
            SweepingLaser(820, 1.6, .0),
            SweepingLaser(620, 2.1, .4),
            SweepingLaser(420, 1.7, .9),
            SweepingLaser(220, 2.3, 1.2),
        ]),

        # 25-26 Five lasers
        Level("Five Right", 80, 920, [
            SweepingLaser(170, 1.8),
            SweepingLaser(340, 1.8, .3),
            SweepingLaser(510, 1.8, .6),
            SweepingLaser(680, 1.8, .9),
            SweepingLaser(850, 1.8, 1.2),
        ]),
        Level("Five Left", 920, 80, [
            SweepingLaser(850, 1.8),
            SweepingLaser(680, 1.8, .3),
            SweepingLaser(510, 1.8, .6),
            SweepingLaser(340, 1.8, .9),
            SweepingLaser(170, 1.8, 1.2),
        ]),

        # 27-28 Fast five
        Level("Sprint Right", 80, 920, [
            SweepingLaser(180, 1.2, grow_speed=800),
            SweepingLaser(340, 1.2, .25, 800),
            SweepingLaser(500, 1.2, .50, 800),
            SweepingLaser(660, 1.2, .75, 800),
            SweepingLaser(820, 1.2, 1.00, 800),
        ]),
        Level("Sprint Left", 920, 80, [
            SweepingLaser(820, 1.2, grow_speed=800),
            SweepingLaser(660, 1.2, .25, 800),
            SweepingLaser(500, 1.2, .50, 800),
            SweepingLaser(340, 1.2, .75, 800),
            SweepingLaser(180, 1.2, 1.00, 800),
        ]),

        # 29-30 Finales
        Level("Final Right", 80, 920, [
            SweepingLaser(180, 1.5, .0, 850),
            SweepingLaser(340, 1.8, .2, 750),
            SweepingLaser(500, 1.6, .6, 900),
            SweepingLaser(660, 1.9, 1.0, 700),
            SweepingLaser(820, 1.4, 1.2, 850),
        ]),

        Level("Final Left", 920, 80, [
            SweepingLaser(820, 1.5, .0, 850),
            SweepingLaser(660, 1.8, .2, 750),
            SweepingLaser(500, 1.6, .6, 900),
            SweepingLaser(340, 1.9, 1.0, 700),
            SweepingLaser(180, 1.4, 1.2, 850),
        ]),
    ]