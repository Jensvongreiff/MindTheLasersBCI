import time


class TrainingTrial:
    def __init__(
        self,
        trial_number,
        ground_truth,
        start_x,
        goal_x,
    ):
        self.trial_number = trial_number
        self.ground_truth = ground_truth
        self.start_x = start_x
        self.goal_x = goal_x

        self.start_time = None

        self.decision_made = False
        self.completed = False
        self.correct = None
        self.accepted_command = None

    def start(self):
        self.start_time = time.perf_counter()

        self.decision_made = False
        self.completed = False
        self.correct = None
        self.accepted_command = None

    def decision_time(self):
        if self.start_time is None:
            return 0.0

        return time.perf_counter() - self.start_time

    def accept_command(self, command):
        if self.decision_made:
            return

        self.accepted_command = command
        self.correct = command == self.ground_truth
        self.decision_made = True