# Global Execution Rules

## Foreground-Only Policy

1. All scripts and commands must run in the foreground.
2. Background execution is forbidden.
3. Do not use `&`, `nohup`, `disown`, `screen`, `tmux`, daemon mode, or async/background task launchers.
4. Monitoring loops must also run in the foreground in the active terminal.
5. If a long-running job is required (download/training/monitoring), keep it attached to the current foreground terminal session.

## Operational Notes

1. Prefer blocking, synchronous execution for all automation.
2. If a process must be restarted after failure, restart it in the foreground only.
3. Do not leave residual background child processes after command completion.
