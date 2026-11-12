from mind_the_lasers.src.game.input_controller import Command
from mind_the_lasers.src.game.settings import WIDTH
from mind_the_lasers.src.game.training_trial import TrainingTrial


class ErrPTrial(TrainingTrial):
    def __init__(self, trial_number, trial_type, scenario):
        ground_truth = {
            "left": Command.LEFT,
            "right": Command.RIGHT,
        }[scenario]
        goal_x = 80 if scenario == "left" else WIDTH - 80

        super().__init__(
            trial_number=trial_number,
            ground_truth=ground_truth,
            start_x=WIDTH // 2,
            goal_x=goal_x,
        )

        self.trial_type = trial_type
        self.scenario = scenario
