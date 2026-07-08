import argparse
import multiprocessing
import queue
import sys

from .pipeline_config import build_pipeline
from .pipeline_constructor import bci_worker_process
from .prediction_sender import PredictionSender
from .signal import (
    LSLStreamer,
    OfflineStreamer,
    EEGDataLoaderOffline,
)
from .smoothing import SmoothingController


def main():

    parser = argparse.ArgumentParser(
        description="Mind the Lasers Pipeline",
    )

    parser.add_argument(
        "--mode",
        choices=["live", "offline"],
        default="live",
    )

    parser.add_argument(
        "--baseline",
        choices=["eegnet", "csp-lda", "bp-lda", "wavelet-lda"],
        default="csp-lda",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Offline .xdf dataset",
    )

    parser.add_argument(
        "--ip",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        default=5005,
        type=int,
    )

    args = parser.parse_args()

    fs = 250

    window_samples = int(fs * 1.0)

    input_queue = multiprocessing.Queue(maxsize=1)

    output_queue = multiprocessing.Queue(maxsize=1)

    pipeline = build_pipeline(
        args.baseline,
        window_samples,
    )

    if args.mode == "offline":

        if args.dataset is None:
            print("Offline mode requires --dataset")
            sys.exit(1)

        print("Loading dataset...")

        data_loader = EEGDataLoaderOffline(data_path=args.dataset)

        X_train, X_test, y_train, y_test = data_loader.load_data()

        pipeline.data_loader = data_loader

        pipeline.calibrate(
            X_train,
            y_train,
        )

        streamer = OfflineStreamer(
            X_test,
            y_test,
            input_queue,
            window_samples,
            fs,
        )

    else:

        streamer = LSLStreamer(
            input_queue,
            window_samples,
            fs=fs,
        )

    worker = multiprocessing.Process(
        target=bci_worker_process,
        args=(
            pipeline,
            input_queue,
            output_queue,
        ),
        daemon=True,
    )

    sender = PredictionSender(
        ip=args.ip,
        port=args.port,
    )

    smoothing = SmoothingController()

    worker.start()

    streamer.start()

    print(
        f"Sending predictions to {args.ip}:{args.port}"
    )

    try:

        while True:

            try:

                msg = output_queue.get(timeout=0.1)

            except queue.Empty:
                continue

            if msg is None:
                print("Pipeline finished.")
                break

            label, confidence, rejected = smoothing.process(
                msg["probabilities"]
            )

            if rejected:
                continue

            sender.send(
                prediction=label,
                confidence=confidence,
            )

            print(
                f"Sent: {label:5s} | conf={confidence:.2f}"
            )

    except KeyboardInterrupt:
        print("Stopping pipeline...")

    finally:

        streamer.stop()

        if args.mode == "live":
            input_queue.put(None)

        worker.join(timeout=2)

        print("Done.")


if __name__ == "__main__":
    main()