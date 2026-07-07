import argparse
import multiprocessing
import sys
import os

from mind_the_lasers.src.game.game import Game
from mind_the_lasers.src.game.input_controller import KeyboardController, BCIController, UnifiedController
from mind_the_lasers.src.pipeline.smoothing import SmoothingController
from mind_the_lasers.src.pipeline.feature_extraction import CSPWrapper
from mind_the_lasers.src.pipeline.model import EEGNetBCIWrapper, LDAWrapper
from mind_the_lasers.src.pipeline.pipeline_constructor import BCIPipeline, bci_worker_process
from mind_the_lasers.src.pipeline.signal import LSLStreamer, OfflineStreamer, load_and_split_offline_data

def setup_pipeline(baseline: str, window_samples: int):
    base_dir = os.path.dirname(__file__)
    
    if baseline == 'eegnet':
        weights_path = os.path.join(base_dir, 'pipeline', 'weights', 'eegnet.pt')
        model = EEGNetBCIWrapper(weights_path=weights_path, n_channels=22, n_samples=window_samples)
        return BCIPipeline(end_to_end_model=model)
    else:
        csp_path = os.path.join(base_dir, 'pipeline', 'weights', 'csp.pkl')
        lda_path = os.path.join(base_dir, 'pipeline', 'weights', 'lda.pkl')
        return BCIPipeline(
            feature_step=CSPWrapper(model_path=csp_path),
            classifier_step=LDAWrapper(model_path=lda_path)
        )

def main():
    parser = argparse.ArgumentParser(description="Mind The Lasers: BCI Integration Runner")
    parser.add_argument('--mode', choices=['live', 'offline'], default='live', help="Data ingestion paradigm.")
    parser.add_argument('--baseline', choices=['eegnet', 'csp-lda'], default='eegnet', help="Underlying classification logic.")
    parser.add_argument('--dataset', type=str, help="Absolute path to BCI2a .mat file (Required if --mode offline).")
    args = parser.parse_args()

    fs = 250
    window_samples = int(fs * 1.0) # 1-second dynamic window processing
    
    input_queue = multiprocessing.Queue(maxsize=1) 
    output_queue = multiprocessing.Queue(maxsize=1)

    pipeline = setup_pipeline(args.baseline, window_samples)
    streamer = None

    if args.mode == 'offline':
        if not args.dataset:
            print("ERROR: --dataset path is strictly required for offline simulation.")
            sys.exit(1)
            
        print(f"Initiating Offline BCI Simulation. Parsing {args.dataset}...")
        X_train, X_test, y_train, y_test = load_and_split_offline_data(args.dataset, fs)
        pipeline.calibrate(X_train, y_train)
        
        streamer = OfflineStreamer(X_test, y_test, input_queue, window_samples, fs)
    else:
        streamer = LSLStreamer(input_queue, window_samples, fs=fs)
        
    worker_proc = multiprocessing.Process(
        target=bci_worker_process, 
        args=(pipeline, input_queue, output_queue),
        daemon=True
    )
    worker_proc.start()
    streamer.start()

    unified_ctrl = UnifiedController(
        bci_controller=BCIController(output_queue, SmoothingController(), baseline_name=args.baseline), 
        keyboard_controller=KeyboardController()
    )

    try:
        print("Launching Mind The Lasers Environment...")
        game = Game(controller=unified_ctrl) 
        game.run()
    finally:
        print("Commencing Graceful Shutdown Protocols...")
        streamer.stop()
        if args.mode == 'live':
            input_queue.put(None) 
        worker_proc.join(timeout=2)
        sys.exit(0)

if __name__ == "__main__":
    main()