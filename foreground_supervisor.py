#!/usr/bin/env python3
import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def latest_checkpoint_step(ckpt_dir: Path) -> int:
    latest = 0
    for path in glob.glob(str(ckpt_dir / "*.pt")):
        base = Path(path).stem
        if base.isdigit():
            latest = max(latest, int(base))
    return latest


def run_cmd(cmd, env=None):
    print(f"[SUPERVISOR] Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env)


def main():
    repo_dir = Path(__file__).resolve().parent
    os.chdir(repo_dir)

    # Foreground-only runtime config
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")

    os.environ.setdefault("HF_HOME", "/host/ssd2/xiong-p/repa/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{os.environ['HF_HOME']}/datasets")

    imagenet_local_dir = Path(env_str("IMAGENET_LOCAL_DIR", "/host/ssd2/xiong-p/repa/imagenet-1k"))
    output_dir = Path(env_str("OUTPUT_DIR", "exps"))
    exp_name = env_str("EXP_NAME", "sit-xl-dinov2-b-repae-hf-imagenet")
    max_train_steps = env_int("MAX_TRAIN_STEPS", 400000)
    checkpointing_steps = env_int("CHECKPOINTING_STEPS", 10000)
    max_restarts = env_int("MAX_RESTARTS", 200)
    stale_seconds = env_int("STALE_SECONDS", 5400)
    base_poll = env_int("BASE_POLL_SECONDS", 60)
    max_poll = env_int("MAX_POLL_SECONDS", 900)
    quad_coef = env_int("POLL_QUAD_COEF", 15)
    label_map_json = env_str("LABEL_MAP_JSON", "")

    output_dir.mkdir(parents=True, exist_ok=True)
    imagenet_local_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)

    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    exp_dir = output_dir / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    runtime_log = exp_dir / "runtime.log"
    watchdog_log = exp_dir / "watchdog.log"
    exp_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with watchdog_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # 1) Required download command in foreground
    log(f"Downloading imagenet-1k to {imagenet_local_dir}")
    dl_cmd = [
        "hf", "download", "ILSVRC/imagenet-1k", "--repo-type", "dataset", "--max-workers", "32",
        "--local-dir", str(imagenet_local_dir)
    ]
    dl_ret = run_cmd(dl_cmd, env=os.environ.copy())
    if dl_ret.returncode != 0:
        log(f"Download failed with code={dl_ret.returncode}")
        return dl_ret.returncode

    # 2) Train with auto-restart and quadratic interval monitoring
    restarts = 0
    while True:
        step = latest_checkpoint_step(ckpt_dir)
        if step >= max_train_steps:
            log(f"Training completed at step={step}")
            return 0

        train_cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", str(gpu_count),
            "--mixed_precision", "bf16",
            "train_repae.py",
            "--max-train-steps", str(max_train_steps),
            "--report-to", "none",
            "--allow-tf32",
            "--mixed-precision", "bf16",
            "--seed", "0",
            "--use-hf-imagenet",
            "--hf-cache-dir", os.environ["HF_DATASETS_CACHE"],
            "--output-dir", str(output_dir),
            "--batch-size", "256",
            "--path-type", "linear",
            "--prediction", "v",
            "--weighting", "uniform",
            "--model", "SiT-XL/2",
            "--checkpointing-steps", str(checkpointing_steps),
            "--loss-cfg-path", "configs/l1_lpips_kl_gan.yaml",
            "--vae", "f8d4",
            "--vae-ckpt", "pretrained/sdvae/sdvae-f8d4.pt",
            "--disc-pretrained-ckpt", "pretrained/sdvae/sdvae-f8d4-discriminator-ckpt.pt",
            "--enc-type", "dinov2-vit-b",
            "--proj-coeff", "0.5",
            "--encoder-depth", "8",
            "--vae-align-proj-coeff", "1.5",
            "--bn-momentum", "0.1",
            "--exp-name", exp_name,
        ]

        if label_map_json:
            train_cmd.extend(["--label-map-json", label_map_json])

        if step > 0:
            train_cmd.extend(["--resume-step", str(step), "--continue-train-exp-dir", str(exp_dir)])

        log(f"Launching training from step={step}, restart={restarts}")
        print("\n" + "="*80)
        print(f"[TRAINING PROCESS OUTPUT BELOW]")
        print("="*80 + "\n", flush=True)

        # Launch training with output directly to console
        with runtime_log.open("a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                train_cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
                bufsize=1
            )

            # Stream output to both console and file in real-time
            last_progress_step = step
            last_progress_ts = time.time()
            no_progress_checks = 0
            terminated_for_stale = False

            while True:
                ret = proc.poll()
                
                # Read available output
                try:
                    line = proc.stdout.readline()
                    if line:
                        print(line, end='', flush=True)
                        logf.write(line)
                        logf.flush()
                except:
                    pass

                if ret is not None:
                    # Drain remaining output
                    remaining = proc.stdout.read()
                    if remaining:
                        print(remaining, end='', flush=True)
                        logf.write(remaining)
                    break

                sleep_s = 0.5
                time.sleep(sleep_s)

                new_step = latest_checkpoint_step(ckpt_dir)
                now = time.time()

                if new_step > last_progress_step:
                    last_progress_step = new_step
                    last_progress_ts = now
                    no_progress_checks = 0
                    msg = f"[SUPERVISOR] Progress checkpoint step={new_step}\n"
                    print(msg, end='', flush=True)
                    logf.write(msg)
                else:
                    no_progress_checks += 1

                if new_step >= max_train_steps:
                    msg = f"[SUPERVISOR] Reached target step={new_step}, stopping training process\n"
                    print(msg, end='', flush=True)
                    logf.write(msg)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return 0

                if now - last_progress_ts > stale_seconds:
                    msg = f"[SUPERVISOR] Stale training detected, terminating current run\n"
                    print(msg, end='', flush=True)
                    logf.write(msg)
                    terminated_for_stale = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

            exit_code = proc.returncode

        print("\n" + "="*80)
        print(f"[TRAINING PROCESS EXITED WITH CODE {exit_code}]")
        print("="*80 + "\n", flush=True)

        step = latest_checkpoint_step(ckpt_dir)
        if step >= max_train_steps:
            log(f"Training completed at step={step}")
            return 0

        restarts += 1
        if restarts > max_restarts:
            log(f"Hit MAX_RESTARTS={max_restarts}, aborting")
            return 1

        if exit_code != 0:
            log(f"ERROR: Training exited with code={exit_code}, latest_step={step}")
            log(f"Waiting for user to fix the issue...")
            time.sleep(15)

        log(f"Restart {restarts}/{max_restarts}, resuming from step={step}")
        time.sleep(5)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SUPERVISOR] Interrupted by user", flush=True)
        sys.exit(130)
