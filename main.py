import sys
import os
import argparse
import json


def build_argv_from_namespace(ns: argparse.Namespace):
    argv = ["train.py"]
    for k, v in vars(ns).items():
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(v))
    return argv


def main():
    parser = argparse.ArgumentParser(description="Launcher for train.py with sensible defaults.")
    parser.add_argument("--train_manifest", type=str, default=None, help="Path to training manifest CSV")
    parser.add_argument("--val_manifest", type=str, default=None, help="Path to validation manifest CSV")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--text_model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--speech_model_name", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP (CUDA only)")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--max_audio_seconds", type=float, default=10.0)
    parser.add_argument("--cache_audio", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    args, unknown = parser.parse_known_args()

    # Basic validation: require manifests
    if args.train_manifest is None or args.val_manifest is None:
        print("Error: --train_manifest and --val_manifest must be provided.")
        parser.print_help()
        return

    # Build argv and call train.main()
    argv = build_argv_from_namespace(args)
    # include any unknown args passed through
    if unknown:
        argv.extend(unknown)

    print("Launching training with arguments:", " ".join(argv[1:]))
    import sys
    sys.argv = argv
    import train
    train.main()


if __name__ == "__main__":
    main()