import multiprocessing
import sys
import os

from mind_the_lasers.src.game.game import Game
from mind_the_lasers.src.game.input_controller import KeyboardController, BCIController, UnifiedController
from mind_the_lasers.src.pipeline.smoothing import SmoothingController
from mind_the_lasers.src.pipeline.model import EEGNetBCIWrapper
from mind_the_lasers.src.pipeline.pipeline_constructor import BCIPipeline, bci_worker_process
from mind_the_lasers.src.pipeline.signal import LSLStreamer

def main():
    fs = 250
    window_samples = fs * 1 
    n_channels = 22 
    weights_path = os.path.join(os.path.dirname(__file__), 'pipeline', 'eegnet_weights.pt')

    # 1. Pipeline Initialization
    eegnet_wrapper = EEGNetBCIWrapper(
        weights_path=weights_path, 
        n_channels=n_channels, 
        n_samples=window_samples
    )
    pipeline = BCIPipeline(end_to_end_model=eegnet_wrapper)

    # 2. Optional Calibration Phase
    if not os.path.exists(weights_path) or "--calibrate" in sys.argv:
        pipeline.calibrate(trials_per_class=15, trial_duration=1.0, fs=fs)

    # 3. Multiprocessing Initialization
    input_queue = multiprocessing.Queue(maxsize=1) 
    output_queue = multiprocessing.Queue(maxsize=1)

    # 4. Spawn Subprocesses & Threads
    worker_proc = multiprocessing.Process(
        target=bci_worker_process, 
        args=(pipeline, input_queue, output_queue),
        daemon=True
    )
    worker_proc.start()

    stream_thread = LSLStreamer(input_queue=input_queue, window_samples=window_samples)
    stream_thread.start()

    # 5. Game Controller Initialization
    smoothing = SmoothingController(window_size=5, confidence_threshold=0.6)
    bci_ctrl = BCIController(output_queue=output_queue, smoothing_controller=smoothing)
    kb_ctrl = KeyboardController()
    unified_ctrl = UnifiedController(bci_controller=bci_ctrl, keyboard_controller=kb_ctrl)

    # 6. Execute Game Loop
    try:
        print("Launching Mind The Lasers...")
        game = Game(controller=unified_ctrl) 
        game.run()
    finally:
        print("Shutting down processes...")
        stream_thread.stop()
        input_queue.put(None) 
        worker_proc.join(timeout=2)
        sys.exit(0)

if __name__ == "__main__":
    main()