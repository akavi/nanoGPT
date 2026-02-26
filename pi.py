#!/usr/bin/env python3
"""Prime Intellect GPU pod management CLI for nanoGPT."""

import argparse
import datetime
import json
import os
import signal
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
QUEUE_FILE = "~/.pi_queue"

# The queue runner loop that runs inside tmux on the remote.
# It processes commands from QUEUE_FILE one at a time.
QUEUE_RUNNER = f"""\
while true; do
  if [ -s {QUEUE_FILE} ]; then
    cmd=$(head -1 {QUEUE_FILE});
    tail -n +2 {QUEUE_FILE} > {QUEUE_FILE}.tmp && mv {QUEUE_FILE}.tmp {QUEUE_FILE};
    echo ">>> $cmd";
    eval "$cmd";
  else
    sleep 2;
  fi;
done
"""


# ── State management ──────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def clear_state():
    """Clear pod-specific state but preserve run tracking data."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Preserve run management fields
        preserved = {}
        for key in ("next_run_id", "last_run_id", "runs", "api_key"):
            if key in state:
                preserved[key] = state[key]
        # Kill any active provisioner since the pod is going away
        if "provisioner_pid" in state:
            try:
                os.kill(state["provisioner_pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Kill any active watcher since the pod is going away
        if "watcher_pid" in state:
            try:
                os.kill(state["watcher_pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if preserved:
            with open(STATE_FILE, "w") as f:
                json.dump(preserved, f, indent=2)
        else:
            os.remove(STATE_FILE)


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
    """Allocate the next sequential run ID and update state."""
    run_id = state.get("next_run_id", 1)
    state["next_run_id"] = run_id + 1
    state["last_run_id"] = run_id
    runs = state.setdefault("runs", {})
    runs[str(run_id)] = {"status": "queued"}
    save_state(state)
    return run_id


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


def _wait_and_shutdown_inner(run_ids):
    """Poll until queue is idle, sync run outputs, then shut down (unless zombified).
    Runs in a forked background process — logs to ~/.pi/wait.log."""
    log = open(os.path.join(STATE_DIR, "wait.log"), "a")
    def _log(msg):
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log.flush()

    state = load_state()
    _log(f"Waiting for runs {run_ids} to finish...")

    for _ in range(720):  # up to 2 hours at 10s intervals
        if queue_is_idle(state):
            break
        time.sleep(10)
        state = load_state()  # refresh in case pod changed
    _log("Queue idle.")

    # Sync all requested runs
    for run_id in run_ids:
        state = load_state()
        _log(f"Syncing run {run_id}...")
        sync_run(state, run_id)
        runs = state.get("runs", {})
        if str(run_id) in runs:
            runs[str(run_id)]["status"] = "synced"
            save_state(state)

    # Check zombify flag
    state = load_state()
    if not state.get("pod_id"):
        _log("No pod in state — skipping shutdown.")
        log.close()
        return
    r = remote_exec(state, "test -f ~/.pi_zombify", check=False)
    if r.returncode == 0:
        _log("Zombify flag detected — skipping shutdown.")
        log.close()
        return

    # Shut down
    _log("Shutting down pod...")
    pod_id = state["pod_id"]
    r = requests.delete(f"{API_BASE}/pods/{pod_id}", headers=api_headers())
    r.raise_for_status()
    clear_state()
    _log("Pod terminated.")
    log.close()


def kill_existing_watcher(state):
    """Kill any existing watcher process tracked in state."""
    pid = state.get("watcher_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone
        except PermissionError:
            pass


def wait_and_shutdown(state, run_ids):
    """Fork a background process to poll, sync, and shut down.

    Accumulates run_ids across invocations — each call kills the previous
    watcher and starts a new one covering all accumulated runs."""
    # Accumulate run IDs in state
    watcher_run_ids = state.get("watcher_run_ids", [])
    for rid in run_ids:
        if rid not in watcher_run_ids:
            watcher_run_ids.append(rid)
    state["watcher_run_ids"] = watcher_run_ids

    # Kill previous watcher before forking a new one
    kill_existing_watcher(state)

    save_state(state)

    pid = os.fork()
    if pid == 0:
        # Child — detach from terminal
        os.setsid()
        try:
            _wait_and_shutdown_inner(watcher_run_ids)
        except Exception as e:
            with open(os.path.join(STATE_DIR, "wait.log"), "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] Error: {e}\n")
        finally:
            # Clean up watcher state
            try:
                s = load_state()
                s.pop("watcher_pid", None)
                s.pop("watcher_run_ids", None)
                save_state(s)
            except Exception:
                pass
        os._exit(0)
    else:
        # Store watcher PID so next invocation can kill it
        state["watcher_pid"] = pid
        save_state(state)
        print(f"Background watcher started (pid {pid}, runs {watcher_run_ids}). Logs: ~/.pi/wait.log")


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
    """Check if the tmux pane is running something (not just the queue runner shell)."""
    r = remote_exec(state, f"tmux list-panes -t {TMUX_SESSION} -F '#{{pane_current_command}}'", check=False)
    if r.returncode != 0:
        return False
    cmd = r.stdout.strip()
    return cmd not in ("bash", "zsh", "sh", "fish", "sleep", "")


def tmux_capture(state, lines=5):
    """Capture recent tmux output."""
    r = remote_exec(state, f"tmux capture-pane -t {TMUX_SESSION} -p -l {lines}", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


# ── Queue helpers ─────────────────────────────────────────────────────────────

def queue_command(state, cmd):
    """Append a command to the remote queue file."""
    escaped = cmd.replace("'", "'\\''")
    remote_exec(state, f"echo '{escaped}' >> {QUEUE_FILE}")
    # Record locally
    history = state.setdefault("cmd_history", [])
    history.append({"cmd": cmd, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    save_state(state)


def read_queue(state):
    """Read pending commands from the remote queue file."""
    r = remote_exec(state, f"cat {QUEUE_FILE} 2>/dev/null", check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    return [line for line in r.stdout.strip().splitlines() if line.strip()]


def clear_queue(state):
    """Clear the remote queue and interrupt the current command."""
    remote_exec(state, f"truncate -s 0 {QUEUE_FILE}", check=False)
    remote_exec(state, f"tmux send-keys -t {TMUX_SESSION} C-c", check=False)


def queue_is_idle(state):
    """Check if the queue is empty and nothing is actively running."""
    pending = read_queue(state)
    return len(pending) == 0 and not tmux_is_busy(state)


def ensure_runner(state):
    """Ensure the queue runner loop is running in tmux. Start it if not."""
    # Check if tmux session exists
    r = remote_exec(state, f"tmux has-session -t {TMUX_SESSION} 2>/dev/null", check=False)
    if r.returncode != 0:
        remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION}", check=False)

    # If tmux is idle, kill and recreate to clear any stale input (e.g. stuck
    # continuation prompt), then start the runner fresh.
    if not tmux_is_busy(state):
        remote_exec(state, f"tmux kill-session -t {TMUX_SESSION}", check=False)
        remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION}", check=False)
        remote_exec(state, f"touch {QUEUE_FILE}", check=False)
        runner_escaped = QUEUE_RUNNER.replace("'", "'\\''").replace("\n", " ")
        tmux_send(state, runner_escaped)


def enqueue_or_flush(state, cmd, force=False):
    """Queue a command, or flush+interrupt if force is set."""
    ensure_runner(state)
    if force:
        clear_queue(state)
        # Brief pause for Ctrl-C to take effect
        time.sleep(0.5)
    queue_command(state, cmd)
    n = len(read_queue(state))
    if force:
        print(f"Flushed queue. Queued: {cmd}")
    elif n > 1:
        print(f"Queued ({n} pending): {cmd}")
    else:
        print(f"Queued: {cmd}")


# ── Provisioning ──────────────────────────────────────────────────────────────

def _run_when_ready(fn):
    """Fork a background process that waits for provisioning, then calls fn(state).
    If the pod is already ready, calls fn directly without forking."""
    state = load_state()
    if not state.get("provisioning"):
        fn(state)
        return None

    pid = os.fork()
    if pid == 0:
        os.setsid()
        log = open(os.path.join(STATE_DIR, "wait.log"), "a")
        def _log(msg):
            log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            log.flush()
        try:
            _log("Waiting for pod to be ready...")
            for _ in range(180):
                state = load_state()
                if state.get("provision_error"):
                    _log(f"Provisioning failed: {state['provision_error']}")
                    break
                if not state.get("provisioning"):
                    _log("Pod ready — executing deferred job.")
                    fn(state)
                    break
                time.sleep(5)
            else:
                _log("Timed out waiting for pod.")
        except Exception as e:
            _log(f"Deferred job error: {e}")
        log.close()
        os._exit(0)
    return pid


def _provision_pod_inner(pod_id, repo, branch):
    """Background: poll until pod ACTIVE, set up SSH + environment.
    Updates state as it progresses. Logs to ~/.pi/provision.log."""
    log = open(os.path.join(STATE_DIR, "provision.log"), "w")
    def _log(msg):
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log.flush()

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
                state = load_state()
                state.pop("provisioning", None)
                state["provision_error"] = str(s.get("installationFailure"))
                save_state(state)
                log.close()
                return
        elif status in ("ERROR", "TERMINATED"):
            _log(f"Pod entered {status} state.")
            state = load_state()
            state.pop("provisioning", None)
            state["provision_error"] = f"Pod {status}"
            save_state(state)
            log.close()
            return

    if not ip:
        _log("Pod did not become ready in time.")
        state = load_state()
        state.pop("provisioning", None)
        state["provision_error"] = "Timed out waiting for pod"
        save_state(state)
        log.close()
        return

    # Update state with connection info
    state = load_state()
    state["ip"] = ip
    state["port"] = port
    state["ssh_user"] = ssh_user
    save_state(state)
    _log(f"Pod active: {ssh_user}@{ip}:{port}")

    # Wait for SSH
    _log("Waiting for SSH...")
    ssh_ok = False
    for _ in range(30):
        r = remote_exec(state, "echo ok", check=False)
        if r.returncode == 0:
            ssh_ok = True
            break
        time.sleep(3)

    if not ssh_ok:
        _log("SSH never became ready.")
        state = load_state()
        state.pop("provisioning", None)
        state["provision_error"] = "SSH timeout"
        save_state(state)
        log.close()
        return

    # Setup environment
    _log("Setting up environment...")
    remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION} || true")
    remote_exec(state, f"touch {QUEUE_FILE}", check=False)

    setup_cmds = " && ".join([
        f"git clone --depth 1 --branch {branch} {repo}",
        "cd nanoGPT",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        'export PATH="$HOME/.local/bin:$PATH"',
        "uv python install 3.13",
        "uv venv --python 3.13",
        "uv sync",
    ])
    queue_command(state, setup_cmds)

    runner_escaped = QUEUE_RUNNER.replace("'", "'\\''").replace("\n", " ")
    tmux_send(state, runner_escaped)

    # Mark provisioning complete
    state = load_state()
    state.pop("provisioning", None)
    save_state(state)
    _log("Provisioning complete. Queue runner started.")
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

    # Save state immediately so other commands can see the pod
    old_state = load_state()
    state = {}
    for key in ("next_run_id", "last_run_id", "runs", "api_key"):
        if key in old_state:
            state[key] = old_state[key]
    state.update({
        "pod_id": pod_id,
        "name": pod_name,
        "provisioning": True,
    })
    save_state(state)

    # Fork background provisioner
    repo = args.repo
    branch = args.branch
    pid = os.fork()
    if pid == 0:
        os.setsid()
        try:
            _provision_pod_inner(pod_id, repo, branch)
        except Exception as e:
            with open(os.path.join(STATE_DIR, "provision.log"), "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] Fatal error: {e}\n")
            s = load_state()
            s.pop("provisioning", None)
            s["provision_error"] = str(e)
            save_state(s)
        os._exit(0)

    state["provisioner_pid"] = pid
    save_state(state)

    print(f"Pod created: {pod_id}")
    print(f"Provisioning in background (pid {pid}). Logs: ~/.pi/provision.log")
    print(f"You can queue jobs now — they'll start when the pod is ready.")


def cmd_down(args):
    state = require_pod()
    pod_id = state["pod_id"]
    name = state.get("name", pod_id)

    # Kill provisioner if still running
    prov_pid = state.get("provisioner_pid")
    if prov_pid:
        try:
            os.kill(prov_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    if not args.f and not state.get("provisioning"):
        # Wait for queued commands to finish (only if pod is ready)
        pending = read_queue(state)
        if pending or tmux_is_busy(state):
            print(f"Waiting for {len(pending)} queued command(s) to finish...")
            for _ in range(600):  # up to 50 minutes
                if queue_is_idle(state):
                    break
                time.sleep(5)
                pending = read_queue(state)
                print(f"\r  {len(pending)} pending, {'running' if tmux_is_busy(state) else 'idle'}   ",
                      end="", flush=True)
            print()

    print(f"Terminating pod '{name}' ({pod_id})...")
    r = requests.delete(f"{API_BASE}/pods/{pod_id}", headers=api_headers())
    r.raise_for_status()
    clear_state()
    print("Pod terminated.")


def cmd_run(args):
    state = require_pod_ready()
    cmd = " ".join(args.command)
    if not cmd:
        print("Error: No command specified.")
        sys.exit(1)

    enqueue_or_flush(state, cmd, force=args.f)


def cmd_train(args):
    state = require_pod()
    config = args.config_path
    overrides = list(args.overrides) if args.overrides else []
    force = args.f

    # Allocate run ID locally (instant)
    run_id = allocate_run_id(state)
    state["runs"][str(run_id)]["config"] = config
    state["runs"][str(run_id)]["overrides"] = overrides
    save_state(state)

    print(f"Run {run_id}: {config} → outputs/{run_id}/")

    override_str = " ".join(overrides)
    train_cmd = f"cd ~/nanoGPT && uv run {config} --out_dir=outputs/{run_id} {override_str}".strip()

    def _do(state):
        write_run_metadata(state, run_id, config, overrides)
        enqueue_or_flush(state, train_cmd, force=force)
        wait_and_shutdown(state, [run_id])

    pid = _run_when_ready(_do)
    if pid is not None:
        print(f"Job deferred until pod is ready (pid {pid}). Logs: ~/.pi/wait.log")


def cmd_sample(args):
    state = require_pod()
    overrides = list(args.overrides) if args.overrides else []
    force = args.f

    # Look up run
    run_id, run = get_run(state, args.run)
    config = run.get("config")
    if not config:
        print(f"Error: Run {run_id} has no config recorded.")
        sys.exit(1)

    print(f"Sampling run {run_id}: {config} → outputs/{run_id}/")

    override_str = " ".join(overrides)
    sample_cmd = f"cd ~/nanoGPT && uv run {config} --mode=sample --out_dir=outputs/{run_id} {override_str}".strip()

    def _do(state):
        enqueue_or_flush(state, sample_cmd, force=force)
        wait_and_shutdown(state, [run_id])

    pid = _run_when_ready(_do)
    if pid is not None:
        print(f"Job deferred until pod is ready (pid {pid}). Logs: ~/.pi/wait.log")


def cmd_resume(args):
    state = require_pod()
    overrides = list(args.overrides) if args.overrides else []
    force = args.f

    # Look up run
    run_id, run = get_run(state, args.run)
    config = run.get("config")
    if not config:
        print(f"Error: Run {run_id} has no config recorded.")
        sys.exit(1)

    print(f"Resuming run {run_id}: {config} → outputs/{run_id}/")

    override_str = " ".join(overrides)
    resume_cmd = f"cd ~/nanoGPT && uv run {config} --mode=resume --out_dir=outputs/{run_id} {override_str}".strip()

    def _do(state):
        enqueue_or_flush(state, resume_cmd, force=force)
        wait_and_shutdown(state, [run_id])

    pid = _run_when_ready(_do)
    if pid is not None:
        print(f"Job deferred until pod is ready (pid {pid}). Logs: ~/.pi/wait.log")


def cmd_zombify(args):
    state = require_pod_ready()
    remote_exec(state, "touch ~/.pi_zombify")
    print("Zombify flag set — pod will not auto-shutdown after runs.")


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

    if state.get("provisioning"):
        print(f"Local:  provisioning (logs: ~/.pi/provision.log)")
        return
    if state.get("provision_error"):
        print(f"Local:  provision failed — {state['provision_error']}")
        return

    # Check tmux (only if pod is ready)
    if tmux_is_busy(state):
        r = remote_exec(state, f"tmux list-panes -t {TMUX_SESSION} -F '#{{pane_current_command}}'", check=False)
        print(f"Tmux:   running ({r.stdout.strip()})")
    else:
        print(f"Tmux:   idle")

    # Show queued commands
    pending = read_queue(state)
    if pending:
        print(f"\nQueue ({len(pending)} pending):")
        for i, cmd in enumerate(pending, 1):
            print(f"  {i}. {cmd}")

    output = tmux_capture(state, lines=8)
    if output:
        print(f"\nRecent output:\n{output}")


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

    state = load_state()
    state["pods"] = active

    # Auto-select if exactly one active pod and no pod currently selected
    if len(active) == 1 and not state.get("pod_id"):
        select_pod(state, active[0])
        print(f"Auto-selected pod '{state['name']}'.")
    elif len(active) > 1 and not state.get("pod_id"):
        print("Multiple active pods — run `pi list` to select one.")

    save_state(state)

    print(f"Fetched {len(active)} active pod(s) (of {len(pods)} total).")
    return active


def select_pod(state, pod):
    """Set a pod as the currently selected pod in state."""
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
    save_state(state)


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
                    select_pod(state, active[idx])
                    print(f"Selected: {active[idx].get('name', active[idx]['id'])}")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid selection.")


def cmd_history(args):
    state = require_pod()
    history = state.get("cmd_history", [])
    if not history:
        print("No command history.")
        return
    for i, entry in enumerate(history, 1):
        ts = entry.get("ts", "?")
        print(f"  {i}. [{ts}] {entry['cmd']}")


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
    p_train.add_argument("config_path", help="Config file path (e.g. config/face_ard_linear_raster_config.py)")
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

    sub.add_parser("zombify", help="Prevent auto-shutdown after runs")

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
        "upload": cmd_upload,
        "zombify": cmd_zombify,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
