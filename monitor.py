#!/usr/bin/env python3
"""
Simple monitor to check training logs for errors and report them.
Runs in the background and checks the experiment log periodically.
"""
import os
import time
import sys
from pathlib import Path
import subprocess

def check_experiment_log():
    """Check the latest experiment log for errors."""
    exp_dir = Path("exps/student-t-e2e-imagenet-full")
    if not exp_dir.exists():
        return None
    
    log_file = exp_dir / "log.txt"
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Look for error patterns in the last 100 lines
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        error_keywords = ["error", "Error", "ERROR", "exception", "Exception", "FAILED", "failed", "Traceback"]
        
        for line in recent_lines:
            for keyword in error_keywords:
                if keyword in line:
                    return line.strip()
        
        # Check if training is making progress
        if lines:
            last_line = lines[-1].strip()
            # Return last line for progress indication
            if "step" in last_line.lower() or "epoch" in last_line.lower():
                return f"[PROGRESS] {last_line}"
        
        return None
    except Exception as e:
        return f"[MONITOR ERROR] {str(e)}"

def get_ps_info():
    """Get process info to check if training is still running."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split("\n"):
            if "train_repae.py" in line and "python" in line:
                return line.strip()
    except:
        pass
    return None

if __name__ == "__main__":
    check_interval = 30  # Check every 30 seconds
    last_reported = None
    
    print("[MONITOR] Training monitor started. Checking for errors every 30 seconds...")
    
    while True:
        try:
            # Check for errors
            error = check_experiment_log()
            
            if error:
                if error != last_reported:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error}")
                    last_reported = error
            
            # Check if process is still running
            ps_info = get_ps_info()
            if ps_info is None:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Training process not found!")
            
            time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\n[MONITOR] Monitor stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"[MONITOR ERROR] {e}")
            time.sleep(check_interval)
