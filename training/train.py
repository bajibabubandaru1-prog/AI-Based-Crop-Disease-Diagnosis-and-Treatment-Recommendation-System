import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from config import (
    CLASS_NAMES_PATH,
    DEFAULT_MODEL_NAME,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAST_LAYERS,
    MODEL_DIR,
    BATCH_SIZE,
    SUPPORTED_MODEL_NAMES,
    get_model_path,
)
from dataset_loader import load_datasets
from model import build_transfer_learning_model, unfreeze_for_fine_tuning


CHECKPOINT_DIR_NAME = "training_checkpoints"


class ResumeStateCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        state_path: Path,
        phase: str,
        model_name: str,
        weights_arg: str,
        total_epochs: int,
    ):
        super().__init__()
        self.state_path = state_path
        self.phase = phase
        self.model_name = model_name
        self.weights_arg = weights_arg
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self.epoch_seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(
            f"Now running {self.phase} phase epoch {epoch + 1}/{self.total_epochs}"
        )

    def on_epoch_end(self, epoch, logs=None):
        duration = None
        if self.epoch_start_time is not None:
            duration = time.time() - self.epoch_start_time
            self.epoch_seconds.append(duration)
        average_seconds = (
            sum(self.epoch_seconds) / len(self.epoch_seconds)
            if self.epoch_seconds
            else None
        )
        remaining_epochs = max(self.total_epochs - (epoch + 1), 0)
        estimated_remaining_seconds = (
            average_seconds * remaining_epochs if average_seconds is not None else None
        )
        state = {
            "model": self.model_name,
            "weights": self.weights_arg,
            "phase": self.phase,
            "completed_epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "last_epoch_seconds": duration,
            "average_epoch_seconds": average_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "last_metrics": dict(logs or {}),
        }
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if estimated_remaining_seconds is not None:
            print(
                "Approx remaining time: "
                f"{estimated_remaining_seconds / 60:.1f} minutes"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a crop disease classifier.")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="Transfer learning architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Frozen feature-extraction training epochs.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=FINE_TUNE_EPOCHS,
        help="Fine-tuning epochs after unfreezing the final base-model layers.",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use pretrained ImageNet weights or train the base model from scratch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest per-epoch training checkpoint if it exists.",
    )
    parser.add_argument(
        "--continue-from-saved-model",
        action="store_true",
        help="Continue training from the saved model file when no resume checkpoint exists.",
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Epoch already completed when continuing from a saved model without resume state.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Images per training batch. Use 4, 8, or 16 on a slow laptop.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Limit TensorFlow CPU threads. Use 2 or 4 to reduce computer lag.",
    )
    parser.add_argument(
        "--laptop-mode",
        action="store_true",
        help="Use lighter defaults for training while doing other work.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.laptop_mode:
        args.batch_size = min(args.batch_size, 8)
        if args.cpu_threads == 0:
            args.cpu_threads = 2

    if args.cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpu_threads)

    model_path = get_model_path(args.model)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = MODEL_DIR / CHECKPOINT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / f"{args.model}_latest.keras"
    resume_state_path = checkpoint_dir / f"{args.model}_resume_state.json"
    training_log_path = checkpoint_dir / f"{args.model}_training_log.csv"
    resume_state = {}

    prefetch_buffer_size = 1 if args.laptop_mode else None
    train_ds, val_ds, class_names = load_datasets(
        batch_size=args.batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
    )
    weights = None if args.weights == "none" else "imagenet"
    if args.resume and latest_checkpoint_path.exists() and resume_state_path.exists():
        resume_state = json.loads(resume_state_path.read_text(encoding="utf-8"))
        if resume_state.get("model") != args.model:
            raise ValueError(
                f"Resume checkpoint is for {resume_state.get('model')}, not {args.model}."
            )
        if resume_state.get("weights") != args.weights:
            raise ValueError(
                "Resume checkpoint was created with "
                f"--weights {resume_state.get('weights')}. "
                f"Restart with --weights {resume_state.get('weights')} or train fresh."
            )
        print(f"Resuming from checkpoint: {latest_checkpoint_path}")
        model = tf.keras.models.load_model(latest_checkpoint_path)
    elif args.continue_from_saved_model and model_path.exists():
        print(f"Continuing from saved model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        resume_state = {
            "model": args.model,
            "weights": args.weights,
            "phase": "frozen",
            "completed_epoch": args.initial_epoch,
        }
    else:
        if args.resume:
            print("No resume checkpoint found. Starting a fresh training run.")
        if args.continue_from_saved_model:
            print("No saved model found. Starting a fresh training run.")
        try:
            model = build_transfer_learning_model(
                model_name=args.model,
                num_classes=len(class_names),
                weights=weights,
            )
        except Exception as exc:
            if weights != "imagenet":
                raise
            print(f"Could not load pretrained ImageNet weights: {exc}")
            print("Retrying with weights=None. This trains the base model from scratch.")
            weights = None
            model = build_transfer_learning_model(
                model_name=args.model,
                num_classes=len(class_names),
                weights=weights,
            )
            args.weights = "none"

    shared_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            latest_checkpoint_path,
            save_best_only=False,
        ),
        tf.keras.callbacks.CSVLogger(training_log_path, append=True),
        tf.keras.callbacks.EarlyStopping(
            # monitor="val_loss",
            # patience=3,
            # restore_best_weights=True,
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
        ),
    ]

    resume_phase = resume_state.get("phase")
    start_epoch = int(resume_state.get("completed_epoch", 0))
    history = None
    final_history = None

    if resume_phase != "fine_tune" and start_epoch < args.epochs:
        print(
            "Running frozen training phase: "
            f"epoch {start_epoch + 1} to {args.epochs}"
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=start_epoch,
            epochs=args.epochs,
            callbacks=shared_callbacks
            + [
                ResumeStateCallback(
                    resume_state_path,
                    phase="frozen",
                    model_name=args.model,
                    weights_arg=args.weights,
                    total_epochs=args.epochs,
                )
            ],
        )
        final_history = history
    elif resume_phase == "frozen" and start_epoch >= args.epochs:
        print("Frozen training phase already completed in checkpoint.")

    if args.fine_tune_epochs > 0 and weights == "imagenet":
        print(
            "Starting fine-tuning: "
            f"unfreezing last {FINE_TUNE_LAST_LAYERS} {args.model} layers"
        )
        if resume_phase != "fine_tune":
            model = unfreeze_for_fine_tuning(
                model,
                model_name=args.model,
                last_layers=FINE_TUNE_LAST_LAYERS,
            )
            start_epoch = args.epochs
        fine_tune_end_epoch = args.epochs + args.fine_tune_epochs
        if start_epoch < fine_tune_end_epoch:
            print(
                "Running fine-tuning phase: "
                f"epoch {start_epoch + 1} to {fine_tune_end_epoch}"
            )
            fine_tune_history = model.fit(
                train_ds,
                validation_data=val_ds,
                initial_epoch=start_epoch,
                epochs=fine_tune_end_epoch,
                callbacks=shared_callbacks
                + [
                    ResumeStateCallback(
                        resume_state_path,
                    phase="fine_tune",
                    model_name=args.model,
                    weights_arg=args.weights,
                    total_epochs=fine_tune_end_epoch,
                )
            ],
            )
            final_history = fine_tune_history
        else:
            print("Fine-tuning phase already completed in checkpoint.")
            fine_tune_history = None
        if history is not None and fine_tune_history is not None:
            history.history["accuracy"].extend(fine_tune_history.history["accuracy"])
            history.history["val_accuracy"].extend(
                fine_tune_history.history["val_accuracy"]
            )
    elif args.fine_tune_epochs > 0:
        print("Skipping fine-tuning phase because the model is training from scratch.")

    model.save(model_path)
    CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    if final_history is not None:
        print("Final training accuracy:", final_history.history["accuracy"][-1])
        print("Final validation accuracy:", final_history.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()
