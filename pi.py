#!/usr/bin/env python3
"""Prime Intellect GPU pod management CLI for nanoGPT."""

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import signal
import shlex
import subprocess
import sys
import time

import requests

API_BASE = "https://api.primeintellect.ai/api/v1"
STATE_DIR = os.path.expanduser("~/.pi")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
SSH_KEY = os.path.expanduser("~/.ssh/id_primeintellect")
DEFAULT_REPO = "https://github.com/akavi/nanoGPT.git"
DEFAULT_BRANCH = "master"
DEFAULT_GPU_TYPE = "H100_80GB"
DEFAULT_GPU_COUNT = 1
TMUX_SESSION = "pi"
CMD_LOG = "~/.pi_cmd.log"
BUSY_FLAG = "~/.pi_busy"


# ── State management ──────────────────────────────────────────────────────────

LOCK_FILE = os.path.join(STATE_DIR, "state.lock")


@contextlib.contextmanager
def locked_state():
    """Context manager that holds an exclusive lock for the duration.
    Yields the state dict; on exit, writes it back to disk.

    Usage:
        with locked_state() as state:
            state["foo"] = "bar"
            # automatically saved on exit
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    lock_fd = open(LOCK_FILE, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                state = json.load(f)
        else:
            state = {}
        yield state
        # Atomic write: temp file + rename so load_state() never sees a truncated file
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def load_state():
    """Read state without locking. Use locked_state() for read-modify-write."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def clear_state():
    """Clear pod-specific state but preserve run tracking data.
    Resets any 'running' commands to 'pending' (the pod they were running on is gone)."""
    with locked_state() as state:
        daemon_pid = state.get("daemon_pid")
        if daemon_pid:
            try:
                os.kill(daemon_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Reset running commands — the pod is gone
        for cmd in state.get("commands", {}).values():
            if cmd["status"] == "running":
                cmd["status"] = "pending"
        preserved = {k: state[k] for k in ("next_run_id", "last_run_id", "runs", "next_cmd_id", "commands", "api_key") if k in state}
        state.clear()
        state.update(preserved)


def get_api_key():
    key = os.environ.get("PRIME_INTELLECT_API_KEY") or os.environ.get("PRIME_API_KEY")
    if not key:
        state = load_state()
        key = state.get("api_key")
    if not key:
        print("Error: Set PRIME_INTELLECT_API_KEY env var or store api_key in ~/.pi_state.json")
        sys.exit(1)
    return key


def api_headers():
    return {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}


def require_pod():
    state = load_state()
    if not state.get("pod_id"):
        print("Error: No active pod. Run `pi up` first.")
        sys.exit(1)
    return state


def require_pod_or_start():
    """Like require_pod(), but offers to start a pod if none is running."""
    state = load_state()
    if state.get("pod_id"):
        return state
    answer = input("No active pod. Start one? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        sys.exit(1)
    # Build a minimal args namespace with defaults for cmd_up
    up_args = argparse.Namespace(
        repo=DEFAULT_REPO,
        branch=DEFAULT_BRANCH,
        gpu_type=DEFAULT_GPU_TYPE,
        gpu_count=DEFAULT_GPU_COUNT,
    )
    cmd_up(up_args)
    return load_state()


def require_pod_ready():
    """Require an active pod that has finished provisioning."""
    state = load_state()
    if not state.get("pod_id"):
        print("Error: No active pod. Run `pi up` first.")
        sys.exit(1)
    if state.get("provision_error"):
        print(f"Error: Pod provisioning failed: {state['provision_error']}")
        sys.exit(1)
    if state.get("provisioning"):
        print("Waiting for pod to be ready...", end="", flush=True)
        for _ in range(180):  # up to 15 minutes
            time.sleep(5)
            state = load_state()
            if state.get("provision_error"):
                print(f"\nError: Pod provisioning failed: {state['provision_error']}")
                sys.exit(1)
            if not state.get("provisioning"):
                print(" ready.")
                return state
            print(".", end="", flush=True)
        print("\nError: Pod provisioning timed out.")
        sys.exit(1)
    return state


# ── Run management ────────────────────────────────────────────────────────────

def allocate_run_id(state):
    """Allocate the next sequential run ID. Caller must hold locked_state()."""
    run_id = state.get("next_run_id", 1)
    state["next_run_id"] = run_id + 1
    state["last_run_id"] = run_id
    runs = state.setdefault("runs", {})
    runs[str(run_id)] = {}
    return run_id


def allocate_cmd_id(state, cmd_type, cmd, run_id=None, force=False):
    """Allocate a command ID and create a pending command entry. Caller must hold locked_state()."""
    cmd_id = state.get("next_cmd_id", 1)
    state["next_cmd_id"] = cmd_id + 1
    commands = state.setdefault("commands", {})
    entry = {
        "type": cmd_type,
        "cmd": cmd,
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat(),
    }
    if run_id is not None:
        entry["run_id"] = run_id
    if force:
        entry["force"] = True
    commands[str(cmd_id)] = entry
    return cmd_id


def get_run(state, run_id=None):
    """Get run metadata by ID, defaulting to last_run_id."""
    if run_id is None:
        run_id = state.get("last_run_id")
    if run_id is None:
        print("Error: No run ID specified and no previous run.")
        sys.exit(1)
    runs = state.get("runs", {})
    run = runs.get(str(run_id))
    if not run:
        print(f"Error: Run {run_id} not found in local state.")
        sys.exit(1)
    return run_id, run


def get_git_sha():
    """Get current git SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_run_metadata(state, run_id, config, overrides):
    """Write run.json to the remote outputs directory."""
    metadata = {
        "run_id": run_id,
        "config": config,
        "overrides": overrides,
        "git_sha": get_git_sha(),
        "created_at": datetime.datetime.now().isoformat(),
    }
    meta_json = json.dumps(metadata)
    escaped = meta_json.replace("'", "'\\''")
    remote_exec(state, f"mkdir -p ~/nanoGPT/outputs/{run_id}")
    remote_exec(state, f"echo '{escaped}' > ~/nanoGPT/outputs/{run_id}/run.json")


def sync_run(state, run_id):
    """Rsync outputs/$run_id/ from remote to local."""
    user = state.get("ssh_user", "root")
    ip = state["ip"]
    port = state.get("port", 22)
    remote_path = f"{user}@{ip}:~/nanoGPT/outputs/{run_id}/"
    local_path = f"outputs/{run_id}/"
    os.makedirs(local_path, exist_ok=True)
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {SSH_KEY} -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
        remote_path, local_path,
    ]
    print(f"Syncing outputs/{run_id}/ from remote...")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"Synced to {local_path}")
    else:
        print(f"Warning: rsync failed with code {result.returncode}")


def upload_run(state, run_id):
    """Rsync outputs/$run_id/ from local to remote (inverse of sync_run)."""
    user = state.get("ssh_user", "root")
    ip = state["ip"]
    port = state.get("port", 22)
    local_path = f"outputs/{run_id}/"
    remote_path = f"{user}@{ip}:~/nanoGPT/outputs/{run_id}/"
    if not os.path.exists(local_path):
        return False
    remote_exec(state, f"mkdir -p ~/nanoGPT/outputs/{run_id}")
    cmd = [
        "rsync", "-avz",
        "-e", f"ssh -i {SSH_KEY} -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
        local_path, remote_path,
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def sync_cmd_log(state):
    """Rsync the remote command log (~/.pi_cmd.log) to local ~/.pi/cmd.log."""
    user = state.get("ssh_user", "root")
    ip = state["ip"]
    port = state.get("port", 22)
    local_path = os.path.join(STATE_DIR, "cmd.log")
    cmd = [
        "rsync", "-az",
        "-e", f"ssh -i {SSH_KEY} -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
        f"{user}@{ip}:{CMD_LOG}", local_path,
    ]
    subprocess.run(cmd, check=False, capture_output=True)


def check_remote_exit(state):
    """Read the remote command's exit status. Returns exit code as string, or None."""
    r = remote_exec(state, "cat ~/.pi_last_exit 2>/dev/null", check=False)
    code = r.stdout.strip()
    return code if code else None


def _daemon(pod_id, skip_provision=False):
    """Single background daemon: provision → execute jobs → sync → idle/shutdown.
    Forked from cmd_up. Logs to provision.log (provisioning) and wait.log (everything else).
    If skip_provision=True, skip provisioning and go straight to the main loop."""
    log = open(os.path.join(STATE_DIR, "wait.log"), "a")
    def _log(msg):
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log.flush()

    shutdown_requested = False
    def _handle_sigterm(signum, frame):
        nonlocal shutdown_requested
        _log("Received SIGTERM.")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        if not skip_provision:
            _provision_pod_inner(pod_id)

            state = load_state()
            if state.get("provision_error"):
                _log(f"Provisioning failed: {state['provision_error']}")
                return

        _log("Entering main loop.")
        idle_since = None
        IDLE_TIMEOUT = 30

        while not shutdown_requested:
            # Phase 1: Check for completed commands (save state before starting new ones)
            with locked_state() as s:
                s["daemon_heartbeat"] = time.time()

                if not s.get("pod_id"):
                    _log("No pod in state — exiting.")
                    break

                sync_cmd_log(s)

                commands = s.get("commands", {})
                running_id = None
                running_entry = None
                for cid in sorted(commands.keys(), key=int):
                    if commands[cid]["status"] == "running":
                        running_id = cid
                        running_entry = commands[cid]
                        break

                if running_entry and not tmux_is_busy(s):
                    exit_code = check_remote_exit(s)
                    failed = exit_code and exit_code != "0"
                    if failed:
                        _log(f"Command {running_id} failed with exit code {exit_code}")
                        output = tmux_capture(s, lines=30)
                        if output:
                            _log(f"Recent output:\n{output}")

                    run_id = running_entry.get("run_id")
                    if run_id:
                        _log(f"Syncing run {run_id}...")
                        sync_run(s, run_id)

                    running_entry["status"] = "failed" if failed else "completed"
                    _log(f"Command {running_id} {'failed' if failed else 'completed'}.")
                # State saved here — completion is persisted even if next phase fails

            # Phase 2: Start next pending command (separate state block)
            with locked_state() as s:
                if not s.get("pod_id"):
                    break

                commands = s.get("commands", {})
                has_running = any(c["status"] == "running" for c in commands.values())

                if not has_running:
                    for cid in sorted(commands.keys(), key=int):
                        if commands[cid]["status"] == "pending":
                            cmd_entry = commands[cid]
                            _log(f"Executing command {cid}: {cmd_entry.get('type', '?')} (run {cmd_entry.get('run_id', '-')})")

                            # Auto-upload local artifacts for sample/resume
                            run_id = cmd_entry.get("run_id")
                            if run_id and cmd_entry.get("type") in ("sample", "resume"):
                                r = remote_exec(s, f"test -f ~/nanoGPT/outputs/{run_id}/ckpt.pt", check=False)
                                if r.returncode != 0:
                                    local_ckpt = f"outputs/{run_id}/ckpt.pt"
                                    if os.path.exists(local_ckpt):
                                        _log(f"Uploading local outputs/{run_id}/ to remote...")
                                        if upload_run(s, run_id):
                                            _log(f"Upload complete.")
                                        else:
                                            _log(f"Warning: upload_run failed for run {run_id}")
                                    else:
                                        _log(f"Warning: no checkpoint on remote or locally for run {run_id}")

                            _execute_command(s, cmd_entry)
                            cmd_entry["status"] = "running"
                            idle_since = None
                            break
                    else:
                        # No running, no pending — idle
                        if idle_since is None:
                            idle_since = time.time()
                            _log("Queue idle, waiting for new commands...")

                        # Check zombify flag
                        r = remote_exec(s, "test -f ~/.pi_zombify", check=False)
                        if r.returncode == 0:
                            idle_since = None  # stay alive indefinitely
                        elif time.time() - idle_since >= IDLE_TIMEOUT:
                            _log("Idle timeout — shutting down pod...")
                            pod_id_to_delete = s["pod_id"]
                            for cmd in s.get("commands", {}).values():
                                if cmd["status"] == "running":
                                    cmd["status"] = "pending"
                            preserved = {k: s[k] for k in ("next_run_id", "last_run_id", "runs", "next_cmd_id", "commands", "api_key") if k in s}
                            s.clear()
                            s.update(preserved)
                            r = requests.delete(f"{API_BASE}/pods/{pod_id_to_delete}", headers=api_headers())
                            r.raise_for_status()
                            _log("Pod terminated.")
                            break
                else:
                    idle_since = None

            time.sleep(10)

    except Exception as e:
        _log(f"Daemon error: {e}")
    finally:
        with locked_state() as s:
            s.pop("daemon_pid", None)
        log.close()


# ── SSH helpers ───────────────────────────────────────────────────────────────

def ssh_opts(state):
    port = state.get("port", 22)
    user = state.get("ssh_user", "root")
    ip = state["ip"]
    return [
        "ssh",
        "-i", SSH_KEY,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        f"{user}@{ip}",
    ]


def scp_opts(state):
    port = state.get("port", 22)
    return [
        "scp",
        "-i", SSH_KEY,
        "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]


def remote_exec(state, cmd, check=True):
    """Run a command on the remote host (outside tmux)."""
    full = ssh_opts(state) + [cmd]
    return subprocess.run(full, capture_output=True, text=True, check=check)


def tmux_send(state, cmd):
    """Send a command to the remote tmux session."""
    # Escape single quotes in cmd for the outer ssh shell
    escaped = cmd.replace("'", "'\\''")
    remote_exec(state, f"tmux send-keys -t {TMUX_SESSION} '{escaped}' Enter")


def tmux_is_busy(state):
    """Check if a command is actively running in tmux."""
    r = remote_exec(state, f"test -f {BUSY_FLAG}", check=False)
    return r.returncode == 0


def tmux_capture(state, lines=5):
    """Capture recent tmux output."""
    r = remote_exec(state, f"tmux capture-pane -t {TMUX_SESSION} -p -l {lines}", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


# ── Provisioning ──────────────────────────────────────────────────────────────

def _execute_command(state, cmd_entry):
    """Execute a single command — write run metadata if needed, send directly to tmux."""
    run_id = cmd_entry.get("run_id")
    if run_id and cmd_entry["type"] in ("train", "resume"):
        runs = state.get("runs", {})
        run = runs.get(str(run_id), {})
        write_run_metadata(state, run_id, run.get("config", ""), run.get("overrides", []))

    cmd = cmd_entry["cmd"]
    force = cmd_entry.get("force", False)
    if force:
        remote_exec(state, f"tmux send-keys -t {TMUX_SESSION} C-c", check=False)
        time.sleep(0.5)

    # Ensure tmux session exists
    remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION} || true")

    # Send command to tmux, tee to log, write exit code
    separator = '"' * 80
    header = f'printf \'\\n{separator}\\n%s\\n{separator}\\n\' {shlex.quote(cmd)} >> {CMD_LOG}'
    wrapped = f'touch {BUSY_FLAG}; {header}; eval {shlex.quote(cmd)} 2>&1 | tee -a {CMD_LOG}; echo ${{PIPESTATUS[0]}} > ~/.pi_last_exit; rm -f {BUSY_FLAG}'
    tmux_send(state, wrapped)


def _daemon_is_alive():
    """Check if the daemon process is running."""
    state = load_state()
    daemon_pid = state.get("daemon_pid")
    if not daemon_pid:
        return False
    try:
        os.kill(daemon_pid, 0)
        return True
    except ProcessLookupError:
        return False


def _fork_daemon(pod_id, skip_provision=False):
    """Fork a background daemon process. Returns the child pid."""
    pid = os.fork()
    if pid == 0:
        os.setsid()
        daemon_log = open(os.path.join(STATE_DIR, "wait.log"), "a")
        os.dup2(daemon_log.fileno(), 1)
        os.dup2(daemon_log.fileno(), 2)
        try:
            _daemon(pod_id, skip_provision=skip_provision)
        except Exception as e:
            with open(os.path.join(STATE_DIR, "wait.log"), "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] Daemon fatal error: {e}\n")
            with locked_state() as s:
                s.pop("provisioning", None)
                s.pop("daemon_pid", None)
                if not skip_provision:
                    s["provision_error"] = str(e)
        os._exit(0)

    with locked_state() as s:
        s["daemon_pid"] = pid
    return pid


def _ensure_daemon():
    """If no daemon is running, start one for the current pod."""
    if _daemon_is_alive():
        return
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        return
    pid = _fork_daemon(pod_id, skip_provision=True)
    print(f"Started daemon (pid {pid}).")


def _enqueue_command(cmd_type, cmd, run_id=None, force=False):
    """Create a pending command in state. Ensures daemon is running to execute it."""
    with locked_state() as s:
        cmd_id = allocate_cmd_id(s, cmd_type, cmd, run_id=run_id, force=force)

    _ensure_daemon()
    print(f"Command {cmd_id} queued.")


def _provision_pod_inner(pod_id):
    """Background: poll until pod ACTIVE, set up SSH + tmux.
    Updates state as it progresses. Logs to ~/.pi/provision.log."""
    log = open(os.path.join(STATE_DIR, "provision.log"), "a")
    def _log(msg):
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log.flush()

    def _fail(error):
        with locked_state() as state:
            state.pop("provisioning", None)
            state["provision_error"] = error
        log.close()

    _log(f"Provisioning pod {pod_id}...")

    ip = None
    port = 22
    ssh_user = "root"

    for _ in range(120):  # up to 10 minutes
        time.sleep(5)
        try:
            r = requests.get(
                f"{API_BASE}/pods/status",
                headers=api_headers(),
                params={"pod_ids": pod_id},
            )
            r.raise_for_status()
            statuses = r.json().get("data", [])
        except Exception as e:
            _log(f"API error: {e}")
            continue

        if not statuses:
            continue
        s = statuses[0]
        status = s.get("status", "")
        _log(f"Status: {status}, install: {s.get('installationStatus', '?')}")

        if status == "ACTIVE":
            ip = s.get("ip")
            ssh_conn = s.get("sshConnection", "")
            if isinstance(ssh_conn, str) and ssh_conn:
                parts = ssh_conn.split()
                for i, p in enumerate(parts):
                    if "@" in p and not p.startswith("-"):
                        user_ip = p.split("@")
                        ssh_user = user_ip[0]
                        ip = user_ip[1]
                    if p == "-p" and i + 1 < len(parts):
                        port = int(parts[i + 1])
            if isinstance(ip, list):
                ip = ip[0]
            inst = s.get("installationStatus")
            if inst in ("FINISHED", None):
                break
            if inst == "ERROR":
                _log(f"Installation error: {s.get('installationFailure')}")
                _fail(str(s.get("installationFailure")))
                return
        elif status in ("ERROR", "TERMINATED"):
            _log(f"Pod entered {status} state.")
            _fail(f"Pod {status}")
            return

    if not ip:
        _log("Pod did not become ready in time.")
        _fail("Timed out waiting for pod")
        return

    # Update state with connection info
    with locked_state() as state:
        state["ip"] = ip
        state["port"] = port
        state["ssh_user"] = ssh_user
    _log(f"Pod active: {ssh_user}@{ip}:{port}")

    # Wait for SSH (use state snapshot for SSH info)
    _log("Waiting for SSH...")
    state = load_state()
    ssh_ok = False
    for _ in range(30):
        r = remote_exec(state, "echo ok", check=False)
        if r.returncode == 0:
            ssh_ok = True
            break
        time.sleep(3)

    if not ssh_ok:
        _log("SSH never became ready.")
        _fail("SSH timeout")
        return

    # Set up tmux session
    _log("Setting up tmux...")
    remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION} || true")

    # Mark provisioning complete — daemon main loop will pick up pending commands
    with locked_state() as state:
        state.pop("provisioning", None)
    _log("Provisioning complete.")
    log.close()


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_up(args):
    print(f"Checking GPU availability ({args.gpu_type} x{args.gpu_count})...")
    r = requests.get(
        f"{API_BASE}/availability/gpus",
        headers=api_headers(),
        params={"gpu_type": args.gpu_type, "gpu_count": args.gpu_count},
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    if not items:
        print(f"No {args.gpu_type} x{args.gpu_count} available right now.")
        sys.exit(1)

    items.sort(key=lambda x: x["prices"]["onDemand"])
    pick = items[0]
    cloud_id = pick["cloudId"]
    provider = pick["provider"]
    socket = pick.get("socket", "SXM5")
    data_center_id = pick.get("dataCenterId") or pick.get("dataCenter")
    print(f"Found: {provider} / {pick.get('region', '?')} @ ${pick['prices']['onDemand']}/hr")

    # Get SSH key ID
    r = requests.get(f"{API_BASE}/ssh_keys/", headers=api_headers())
    r.raise_for_status()
    ssh_keys = r.json().get("data", [])
    ssh_key_id = None
    for k in ssh_keys:
        if k.get("isPrimary"):
            ssh_key_id = k["id"]
            break
    if not ssh_key_id and ssh_keys:
        ssh_key_id = ssh_keys[0]["id"]

    pod_name = f"nanogpt-{int(time.time()) % 100000}"
    print(f"Creating pod '{pod_name}'...")
    body = {
        "pod": {
            "cloudId": cloud_id,
            "gpuType": args.gpu_type,
            "socket": socket,
            "gpuCount": args.gpu_count,
            "name": pod_name,
            **({"dataCenterId": data_center_id} if data_center_id else {}),
        },
        "provider": {"type": provider},
    }
    if ssh_key_id:
        body["pod"]["sshKeyId"] = ssh_key_id

    r = requests.post(f"{API_BASE}/pods/", headers=api_headers(), json=body)
    if not r.ok:
        print(f"API error {r.status_code}: {r.text}")
    r.raise_for_status()
    pod = r.json()
    pod_id = pod["id"]

    # Clear old pod state (kills stale daemon, preserves run tracking)
    # This also resets any 'running' commands back to 'pending'
    clear_state()

    # Check for pending commands from a previous pod
    prev_state = load_state()
    pending_cmds = {
        cid: c for cid, c in prev_state.get("commands", {}).items()
        if c["status"] == "pending" and c["type"] != "setup"
    }
    if pending_cmds:
        print(f"\n{len(pending_cmds)} pending command(s) from previous pod:")
        for cid in sorted(pending_cmds.keys(), key=int):
            c = pending_cmds[cid]
            run_str = f" run {c['run_id']}" if c.get("run_id") else ""
            print(f"  {cid}. {c['type']}{run_str}: {c['cmd'][:80]}")
        choice = input("\nRe-queue? [a]ll / [s]elect / [n]one (default: all): ").strip().lower()
        if choice in ("n", "none"):
            with locked_state() as s:
                for cid in pending_cmds:
                    s["commands"][cid]["status"] = "abandoned"
            print("All pending commands abandoned.")
        elif choice in ("s", "select"):
            keep = input(f"Enter command IDs to keep (comma-separated): ").strip()
            keep_ids = {x.strip() for x in keep.split(",") if x.strip()}
            with locked_state() as s:
                for cid in pending_cmds:
                    if cid not in keep_ids:
                        s["commands"][cid]["status"] = "abandoned"
            kept = len(keep_ids & set(pending_cmds.keys()))
            print(f"Kept {kept}, abandoned {len(pending_cmds) - kept}.")
        else:
            print("All pending commands will be re-queued.")
        print()

    # Build setup command
    if provider.lower() == "datacrunch":
        uv_install = "snap install --classic astral-uv"
    else:
        uv_install = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    setup_cmd = " && ".join([
        f"git clone --depth 1 --branch {args.branch} {args.repo}",
        "cd nanoGPT",
        uv_install,
        'grep -q ".local/bin" ~/.bashrc || echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc',
        'export PATH="$HOME/.local/bin:$PATH"',
        "uv python install 3.13",
        "uv venv --python 3.13",
        "uv sync",
    ])

    with locked_state() as state:
        state.update({
            "pod_id": pod_id,
            "name": pod_name,
            "provider": provider,
            "provisioning": True,
        })
        # Renumber pending commands so setup runs first
        commands = state.get("commands", {})
        pending = {cid: c for cid, c in commands.items() if c["status"] == "pending"}
        for cid in pending:
            del commands[cid]
        allocate_cmd_id(state, "setup", setup_cmd)
        for c in pending.values():
            allocate_cmd_id(state, c["type"], c["cmd"],
                           run_id=c.get("run_id"), force=c.get("force", False))

    # Fork background daemon (handles provisioning + job execution + sync + shutdown)
    pid = _fork_daemon(pod_id)

    print(f"Pod created: {pod_id}")
    print(f"Daemon started (pid {pid}). Logs: ~/.pi/provision.log, ~/.pi/wait.log")
    print(f"You can queue jobs now — they'll start when the pod is ready.")


def cmd_down(args):
    state = require_pod()
    pod_id = state["pod_id"]
    name = state.get("name", pod_id)

    # Kill daemon if running
    daemon_pid = state.get("daemon_pid")
    if daemon_pid:
        try:
            os.kill(daemon_pid, signal.SIGTERM)
            time.sleep(1)
        except (ProcessLookupError, PermissionError):
            pass

    if not args.f and not state.get("provisioning"):
        # Wait for active commands to finish (only if pod is ready)
        commands = state.get("commands", {})
        has_active = any(c["status"] in ("pending", "running") for c in commands.values())
        if has_active or tmux_is_busy(state):
            print("Waiting for active commands to finish...")
            for _ in range(600):  # up to 50 minutes
                if not tmux_is_busy(state):
                    break
                time.sleep(5)
                print(f"\r  {'running' if tmux_is_busy(state) else 'idle'}   ",
                      end="", flush=True)
            print()

    print(f"Terminating pod '{name}' ({pod_id})...")
    r = requests.delete(f"{API_BASE}/pods/{pod_id}", headers=api_headers())
    r.raise_for_status()
    clear_state()
    print("Pod terminated.")


def cmd_reset(args):
    """Kill daemon and clear pod state (keeps run history)."""
    state = load_state()
    daemon_pid = state.get("daemon_pid")
    if daemon_pid:
        try:
            os.kill(daemon_pid, signal.SIGTERM)
            print(f"Killed daemon (pid {daemon_pid})")
        except ProcessLookupError:
            print(f"Daemon already dead (pid {daemon_pid})")
        except PermissionError:
            print(f"Cannot kill daemon (pid {daemon_pid})")
    clear_state()
    print("State reset. Run history preserved.")


def cmd_restart(args):
    """Restart the daemon for the current pod (e.g. after daemon crash)."""
    state = require_pod()
    pod_id = state["pod_id"]

    # Kill stale daemon if any
    daemon_pid = state.get("daemon_pid")
    if daemon_pid:
        try:
            os.kill(daemon_pid, signal.SIGTERM)
            print(f"Killing old daemon (pid {daemon_pid})...")
            # Wait for it to actually die (it may be mid-SSH call)
            for _ in range(30):
                time.sleep(1)
                try:
                    os.kill(daemon_pid, 0)
                except ProcessLookupError:
                    break
            else:
                os.kill(daemon_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Cannot kill old daemon (pid {daemon_pid})")

    pid = _fork_daemon(pod_id, skip_provision=True)
    print(f"Daemon restarted (pid {pid}). Pending commands will resume.")


def cmd_run(args):
    require_pod_or_start()
    cmd = " ".join(args.command)
    if not cmd:
        print("Error: No command specified.")
        sys.exit(1)

    _enqueue_command("run", cmd, force=args.f)


def cmd_train(args):
    require_pod_or_start()
    overrides = list(args.overrides) if args.overrides else []

    if args.template is not None:
        state = load_state()
        _, template_run = get_run(state, args.template)
        config = template_run.get("config")
        if not config:
            print(f"Error: Template run {args.template} has no config recorded.")
            sys.exit(1)
        # Template overrides first, then CLI overrides on top
        overrides = template_run.get("overrides", []) + overrides
        if args.config_path:
            # If config_path was also given, it was probably an override
            overrides = [args.config_path] + overrides
    else:
        if not args.config_path:
            print("Error: config_path is required (or use --template).")
            sys.exit(1)
        config = args.config_path

    override_str = " ".join(shlex.quote(o) for o in overrides)
    with locked_state() as s:
        run_id = allocate_run_id(s)
        s["runs"][str(run_id)]["config"] = config
        s["runs"][str(run_id)]["overrides"] = overrides

    train_cmd = f"cd ~/nanoGPT && uv run {config} --out_dir=outputs/{run_id} {override_str}".strip()

    print(f"Run {run_id}: {config} → outputs/{run_id}/")
    _enqueue_command("train", train_cmd, run_id=run_id, force=args.f)


def cmd_sample(args):
    state = require_pod_or_start()
    overrides = list(args.overrides) if args.overrides else []

    run_id, run = get_run(state, args.run)
    config = run.get("config")
    if not config:
        print(f"Error: Run {run_id} has no config recorded.")
        sys.exit(1)

    override_str = " ".join(shlex.quote(o) for o in overrides)
    sample_cmd = f"cd ~/nanoGPT && uv run {config} --mode=sample --out_dir=outputs/{run_id} {override_str}".strip()

    print(f"Sampling run {run_id}: {config} → outputs/{run_id}/")
    _enqueue_command("sample", sample_cmd, run_id=run_id, force=args.f)


def cmd_resume(args):
    state = require_pod_or_start()
    overrides = list(args.overrides) if args.overrides else []

    run_id, run = get_run(state, args.run)
    config = run.get("config")
    if not config:
        print(f"Error: Run {run_id} has no config recorded.")
        sys.exit(1)

    override_str = " ".join(shlex.quote(o) for o in overrides)
    resume_cmd = f"cd ~/nanoGPT && uv run {config} --mode=resume --out_dir=outputs/{run_id} {override_str}".strip()

    print(f"Resuming run {run_id}: {config} → outputs/{run_id}/")
    _enqueue_command("resume", resume_cmd, run_id=run_id, force=args.f)


def cmd_sweep(args):
    """Enqueue train + sample for the cartesian product of comma-separated overrides."""
    import itertools

    require_pod_or_start()
    config = args.config_path
    overrides = list(args.overrides) if args.overrides else []

    # Partition overrides into sweep (has commas) vs fixed
    fixed = []
    sweep_keys = []
    sweep_values = []
    for ov in overrides:
        if "=" in ov and "," in ov.split("=", 1)[1]:
            key, vals = ov.split("=", 1)
            sweep_keys.append(key)
            sweep_values.append(vals.split(","))
        else:
            fixed.append(ov)

    if not sweep_keys:
        print("No comma-separated overrides found — nothing to sweep.")
        sys.exit(1)

    combos = list(itertools.product(*sweep_values))
    print(f"Sweep: {len(combos)} runs from {' × '.join(f'{k}={{{",".join(v)}}}' for k, v in zip(sweep_keys, sweep_values))}")

    for combo in combos:
        combo_overrides = fixed + [f"{k}={v}" for k, v in zip(sweep_keys, combo)]
        override_str = " ".join(shlex.quote(o) for o in combo_overrides)

        with locked_state() as s:
            run_id = allocate_run_id(s)
            s["runs"][str(run_id)]["config"] = config
            s["runs"][str(run_id)]["overrides"] = combo_overrides

        train_cmd = f"cd ~/nanoGPT && uv run {config} --out_dir=outputs/{run_id} {override_str}".strip()
        sample_cmd = f"cd ~/nanoGPT && uv run {config} --mode=sample --out_dir=outputs/{run_id} {override_str}".strip()

        label = " ".join(f"{k}={v}" for k, v in zip(sweep_keys, combo))
        print(f"  Run {run_id}: {label}")
        _enqueue_command("train", train_cmd, run_id=run_id)
        _enqueue_command("sample", sample_cmd, run_id=run_id)


def cmd_pull(args):
    require_pod()
    cmd = "cd ~/nanoGPT && git fetch && git reset origin/master --hard"
    _enqueue_command("run", cmd, force=args.f)
    print("Queued git pull (fetch + reset to origin/master).")


def cmd_zombify(args):
    state = require_pod_ready()
    remote_exec(state, "touch ~/.pi_zombify")
    print("Zombify flag set — pod will not auto-shutdown after runs.")


def cmd_mortalize(args):
    state = require_pod_ready()
    remote_exec(state, "rm -f ~/.pi_zombify")
    print("Zombify flag removed — pod will auto-shutdown when idle.")


def cmd_fetch(args):
    state = require_pod_ready()
    # Support --run to fetch by run ID
    if args.run:
        run_id, _ = get_run(state, args.run)
        sync_run(state, run_id)
        return
    out_dir = args.out_dir or state.get("last_out_dir")
    if not out_dir:
        print("Error: No --out-dir or --run specified and no previous train command to infer from.")
        sys.exit(1)

    local_dir = args.local_dir or "."
    os.makedirs(local_dir, exist_ok=True)

    user = state.get("ssh_user", "root")
    ip = state["ip"]
    remote_base = f"{user}@{ip}:~/nanoGPT/{out_dir}"

    if not args.images and not args.checkpoint:
        args.images = True  # default to fetching images

    if args.images:
        print(f"Fetching images from {out_dir}...")
        cmd = scp_opts(state) + ["-r", f"{remote_base}/*.png", f"{local_dir}/"]
        # scp doesn't glob well, use ssh + tar instead
        tar_cmd = f"cd ~/nanoGPT && tar cf - {out_dir}/*.png 2>/dev/null"
        ssh_cmd = ssh_opts(state) + [tar_cmd]
        tar_extract = ["tar", "xf", "-", "-C", local_dir, "--strip-components=1"]
        try:
            p1 = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE)
            p2 = subprocess.Popen(tar_extract, stdin=p1.stdout)
            if p1.stdout:
                p1.stdout.close()
            p2.communicate()
            if p2.returncode == 0:
                print(f"Images saved to {local_dir}/")
            else:
                # Fallback: try scp directly
                subprocess.run(
                    scp_opts(state) + ["-r", f"{remote_base}/", f"{local_dir}/"],
                    check=True,
                )
                print(f"Directory saved to {local_dir}/")
        except subprocess.CalledProcessError as e:
            print(f"Fetch failed: {e}")

    if args.checkpoint:
        print(f"Fetching checkpoint from {out_dir}...")
        cmd = scp_opts(state) + [f"{remote_base}/ckpt.pt", f"{local_dir}/"]
        try:
            subprocess.run(cmd, check=True)
            print(f"Checkpoint saved to {local_dir}/ckpt.pt")
        except subprocess.CalledProcessError as e:
            print(f"Fetch failed: {e}")


def cmd_status(args):
    state = require_pod()
    pod_id = state["pod_id"]

    # Check pod status via API
    r = requests.get(
        f"{API_BASE}/pods/status",
        headers=api_headers(),
        params={"pod_ids": pod_id},
    )
    r.raise_for_status()
    statuses = r.json().get("data", [])

    if statuses:
        s = statuses[0]
        print(f"Pod:    {state.get('name', pod_id)}")
        print(f"Status: {s.get('status', '?')}")
        if state.get("ip"):
            print(f"IP:     {state['ip']}:{state.get('port', 22)}")
        inst = s.get("installationStatus")
        if inst:
            prog = s.get("installationProgress", "")
            print(f"Setup:  {inst}" + (f" ({prog}%)" if prog else ""))
    else:
        print(f"Pod {pod_id}: no status available")
        return

    # Daemon status
    daemon_pid = state.get("daemon_pid")
    heartbeat = state.get("daemon_heartbeat")
    if daemon_pid:
        try:
            os.kill(daemon_pid, 0)
            ago = f", last poll {int(time.time() - heartbeat)}s ago" if heartbeat else ""
            print(f"Daemon: running (pid {daemon_pid}{ago})")
        except ProcessLookupError:
            print(f"Daemon: dead (stale pid {daemon_pid})")
    else:
        print(f"Daemon: not running")

    if state.get("provisioning"):
        print(f"Local:  provisioning (logs: ~/.pi/provision.log)")
        return
    if state.get("provision_error"):
        print(f"Local:  provision failed — {state['provision_error']}")
        return

    # Commands
    commands = state.get("commands", {})
    active = {cid: c for cid, c in commands.items() if c["status"] in ("pending", "running")}
    if active:
        print(f"\nCommands ({len(active)} active):")
        for cid in sorted(active.keys(), key=int):
            c = active[cid]
            run_str = f" run {c['run_id']}" if c.get("run_id") else ""
            print(f"  {cid}. [{c['status']}] {c['type']}{run_str}: {c['cmd'][:80]}")

    # Check tmux (only if pod is ready)
    if tmux_is_busy(state):
        print(f"Tmux:   running")
    else:
        print(f"Tmux:   idle")

    output = tmux_capture(state, lines=8)
    if output:
        print(f"\nRecent output:\n{output}")


def cmd_log(args):
    """Show the remote command log (synced to ~/.pi/cmd.log)."""
    state = require_pod_ready()
    sync_cmd_log(state)
    log_path = os.path.join(STATE_DIR, "cmd.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        n = args.n if hasattr(args, "n") and args.n else 50
        for line in lines[-n:]:
            print(line, end="")
    else:
        print("No command log yet.")


def cmd_sync(args):
    """Fetch active pods from the API and store in state."""
    r = requests.get(
        f"{API_BASE}/pods/",
        headers=api_headers(),
        params={"limit": 100},
    )
    r.raise_for_status()
    data = r.json()
    pods = data.get("data", [])
    active = [p for p in pods if p.get("status") == "ACTIVE"]

    with locked_state() as state:
        state["pods"] = active
        if len(active) == 1 and not state.get("pod_id"):
            select_pod(state, active[0])
            print(f"Auto-selected pod '{state['name']}'.")
        elif len(active) > 1 and not state.get("pod_id"):
            print("Multiple active pods — run `pi list` to select one.")

    print(f"Fetched {len(active)} active pod(s) (of {len(pods)} total).")
    return active


def select_pod(state, pod):
    """Set a pod as the currently selected pod in state. Caller must hold locked_state()."""
    state["pod_id"] = pod["id"]
    state["name"] = pod.get("name", pod["id"])
    ssh_conn = pod.get("sshConnection", "")
    if isinstance(ssh_conn, str) and ssh_conn:
        parts = ssh_conn.split()
        for i, part in enumerate(parts):
            if "@" in part and not part.startswith("-"):
                user_ip = part.split("@")
                state["ssh_user"] = user_ip[0]
                state["ip"] = user_ip[1]
            if part == "-p" and i + 1 < len(parts):
                state["port"] = int(parts[i + 1])
    elif pod.get("ip"):
        state["ip"] = pod["ip"]
        state.setdefault("ssh_user", "root")
        state.setdefault("port", 22)


def cmd_list(args):
    """List active pods."""
    active = cmd_sync(args)
    if not active:
        print("No active pods.")
        return

    state = load_state()
    current_id = state.get("pod_id")

    print()
    for i, p in enumerate(active, 1):
        gpu = p.get("gpuType", "?")
        gpu_count = p.get("gpuCount", "?")
        name = p.get("name", "?")
        pod_id = p.get("id", "?")
        ssh_conn = p.get("sshConnection", "")
        marker = " *" if pod_id == current_id else ""
        print(f"  {i}. {name}  id={pod_id}  {gpu} x{gpu_count}  {ssh_conn or ''}{marker}")

    if len(active) > 1:
        print()
        choice = input(f"Select pod [1-{len(active)}] (enter to keep current): ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(active):
                    with locked_state() as state:
                        select_pod(state, active[idx])
                    print(f"Selected: {active[idx].get('name', active[idx]['id'])}")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid selection.")


def cmd_history(args):
    state = load_state()
    commands = state.get("commands", {})
    if not commands:
        print("No command history.")
        return
    for cid in sorted(commands.keys(), key=int):
        c = commands[cid]
        ts = c.get("created_at", "?")
        run_str = f" run {c['run_id']}" if c.get("run_id") else ""
        print(f"  {cid}. [{c['status']}] {ts} {c['type']}{run_str}: {c['cmd'][:100]}")


def _rm_command(s, cmd_id):
    """Cancel or kill a single command by ID. Caller must hold locked_state()."""
    cmd_id = str(cmd_id)
    commands = s.get("commands", {})
    if cmd_id not in commands:
        print(f"Error: Command {cmd_id} not found.")
        sys.exit(1)
    entry = commands[cmd_id]
    status = entry["status"]
    if status in ("completed", "failed", "cancelled"):
        print(f"Command {cmd_id} already {status}.")
        return
    if status == "running":
        if s.get("pod_id") and s.get("ip"):
            remote_exec(s, f"tmux send-keys -t {TMUX_SESSION} C-c", check=False)
            time.sleep(0.5)
            remote_exec(s, f"rm -f {BUSY_FLAG}", check=False)
        print(f"Killed running command {cmd_id}.")
    else:
        print(f"Cancelled pending command {cmd_id}.")
    entry["status"] = "cancelled"


def cmd_rm(args):
    """Remove a command — cancel if pending, kill if running."""
    if args.cmd_id is not None:
        with locked_state() as s:
            _rm_command(s, args.cmd_id)
        return

    # Interactive mode: show active commands and let user pick
    state = load_state()
    commands = state.get("commands", {})
    active = [(cid, c) for cid, c in sorted(commands.items(), key=lambda x: int(x[0]))
              if c["status"] in ("pending", "running")]
    if not active:
        print("No active commands.")
        return

    for cid, c in active:
        run_str = f" run {c['run_id']}" if c.get("run_id") else ""
        print(f"  {cid}. [{c['status']}] {c['type']}{run_str}: {c['cmd'][:100]}")

    print()
    choice = input("Command ID(s) to remove (comma-separated, or 'all'): ").strip()
    if not choice:
        return
    if choice == "all":
        ids = [cid for cid, _ in active]
    else:
        ids = [x.strip() for x in choice.split(",")]

    with locked_state() as s:
        for cid in ids:
            _rm_command(s, cid)


def cmd_upload(args):
    state = require_pod_ready()
    run_id, _ = get_run(state, args.run)
    local_path = args.path or f"outputs/{run_id}/ckpt.pt"
    if not os.path.exists(local_path):
        print(f"Error: Local file not found: {local_path}")
        sys.exit(1)

    user = state.get("ssh_user", "root")
    ip = state["ip"]
    port = state.get("port", 22)
    remote_dir = f"~/nanoGPT/outputs/{run_id}"
    remote_exec(state, f"mkdir -p {remote_dir}")

    print(f"Uploading {local_path} → {remote_dir}/ckpt.pt ...")
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {SSH_KEY} -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
        local_path, f"{user}@{ip}:{remote_dir}/ckpt.pt",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"Uploaded to outputs/{run_id}/ckpt.pt on remote.")
    else:
        print(f"Upload failed with code {result.returncode}")
        sys.exit(1)


def cmd_ssh(args):
    state = require_pod_ready()
    cmd = ssh_opts(state) + ["-t", f"tmux attach -t {TMUX_SESSION} || tmux new -s {TMUX_SESSION}"]
    os.execvp("ssh", cmd)


# ── CLI setup ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="pi", description="Prime Intellect GPU pod manager")
    sub = parser.add_subparsers(dest="cmd")

    p_up = sub.add_parser("up", help="Spin up a GPU pod")
    p_up.add_argument("--repo", default=DEFAULT_REPO, help="Git repo URL")
    p_up.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch")
    p_up.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="GPU type (e.g. H100_80GB)")
    p_up.add_argument("--gpu-count", type=int, default=DEFAULT_GPU_COUNT, help="Number of GPUs")

    p_down = sub.add_parser("down", help="Terminate the current pod")
    p_down.add_argument("-f", action="store_true", help="Terminate immediately without waiting for queue")

    p_run = sub.add_parser("run", help="Run a command on the remote pod")
    p_run.add_argument("-f", action="store_true", help="Flush queue and run immediately")
    p_run.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("-f", action="store_true", help="Flush queue and run immediately")
    p_train.add_argument("--template", type=int, default=None, help="Run ID to use as template (reuse its config and overrides)")
    p_train.add_argument("config_path", nargs="?", default=None, help="Config file path (e.g. config/face_ard_linear_raster_config.py)")
    p_train.add_argument("overrides", nargs=argparse.REMAINDER, help="Training overrides (e.g. --n_step=1)")

    p_sample = sub.add_parser("sample", help="Sample from a trained model")
    p_sample.add_argument("-f", action="store_true", help="Flush queue and run immediately")
    p_sample.add_argument("--run", type=int, default=None, help="Run ID to sample from (default: last run)")
    p_sample.add_argument("overrides", nargs=argparse.REMAINDER, help="Sampling overrides (e.g. --n_step=1)")

    p_resume = sub.add_parser("resume", help="Resume training from a previous run")
    p_resume.add_argument("-f", action="store_true", help="Flush queue and run immediately")
    p_resume.add_argument("--run", type=int, default=None, help="Run ID to resume (default: last run)")
    p_resume.add_argument("overrides", nargs=argparse.REMAINDER, help="Resume overrides (e.g. --n_step=1)")

    p_fetch = sub.add_parser("fetch", help="Fetch results from the pod")
    p_fetch.add_argument("--images", action="store_true", help="Fetch PNG images")
    p_fetch.add_argument("--checkpoint", action="store_true", help="Fetch checkpoint")
    p_fetch.add_argument("--out-dir", help="Remote output directory name")
    p_fetch.add_argument("--run", type=int, default=None, help="Run ID to fetch")
    p_fetch.add_argument("--local-dir", default=".", help="Local destination directory")

    sub.add_parser("status", help="Check pod and training status")
    sub.add_parser("history", help="Show command history for the current pod")
    sub.add_parser("ssh", help="SSH into the pod tmux session")
    sub.add_parser("sync", help="Fetch active pods from the API")
    sub.add_parser("list", help="List active pods")
    p_upload = sub.add_parser("upload", help="Upload a checkpoint to the remote pod")
    p_upload.add_argument("--run", type=int, default=None, help="Run ID (default: last run)")
    p_upload.add_argument("--path", help="Local checkpoint path (default: outputs/<run>/ckpt.pt)")

    p_log = sub.add_parser("log", help="Show remote command output log")
    p_log.add_argument("-n", type=int, default=50, help="Number of lines to show (default: 50)")

    p_rm = sub.add_parser("rm", help="Remove a pending or running command")
    p_rm.add_argument("cmd_id", nargs="?", type=int, default=None, help="Command ID to remove (interactive if omitted)")
    p_sweep = sub.add_parser("sweep", help="Sweep: train+sample for cartesian product of comma-separated overrides")
    p_sweep.add_argument("config_path", help="Config file path")
    p_sweep.add_argument("overrides", nargs=argparse.REMAINDER, help="Overrides (comma-separated values are swept)")

    p_pull = sub.add_parser("pull", help="Git fetch and reset to origin/master on the pod")
    p_pull.add_argument("-f", action="store_true", help="Flush queue and run immediately")
    sub.add_parser("zombify", help="Prevent auto-shutdown after runs")
    sub.add_parser("mortalize", help="Re-enable auto-shutdown after runs")
    sub.add_parser("restart", help="Restart the daemon for the current pod")
    sub.add_parser("reset", help="Kill daemon and clear pod state (keeps run history)")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "up": cmd_up,
        "down": cmd_down,
        "run": cmd_run,
        "train": cmd_train,
        "sample": cmd_sample,
        "resume": cmd_resume,
        "fetch": cmd_fetch,
        "status": cmd_status,
        "history": cmd_history,
        "ssh": cmd_ssh,
        "sync": cmd_sync,
        "list": cmd_list,
        "rm": cmd_rm,
        "upload": cmd_upload,
        "log": cmd_log,
        "sweep": cmd_sweep,
        "pull": cmd_pull,
        "zombify": cmd_zombify,
        "mortalize": cmd_mortalize,
        "restart": cmd_restart,
        "reset": cmd_reset,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
