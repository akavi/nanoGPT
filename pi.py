#!/usr/bin/env python3
"""Prime Intellect GPU pod management CLI for nanoGPT."""

import argparse
import json
import os
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
    cmd=$(head -1 {QUEUE_FILE})
    tail -n +2 {QUEUE_FILE} > {QUEUE_FILE}.tmp && mv {QUEUE_FILE}.tmp {QUEUE_FILE}
    echo ">>> $cmd"
    eval "$cmd"
  else
    sleep 2
  fi
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
    if os.path.exists(STATE_FILE):
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
    return cmd not in ("bash", "zsh", "sh", "fish", "")


def tmux_capture(state, lines=5):
    """Capture recent tmux output."""
    r = remote_exec(state, f"tmux capture-pane -t {TMUX_SESSION} -p -l {lines}", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


# ── Queue helpers ─────────────────────────────────────────────────────────────

def queue_command(state, cmd):
    """Append a command to the remote queue file."""
    escaped = cmd.replace("'", "'\\''")
    remote_exec(state, f"echo '{escaped}' >> {QUEUE_FILE}")


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

    # Check if the runner is already going — look for the sleep/eval pattern
    r = remote_exec(state, f"tmux capture-pane -t {TMUX_SESSION} -p -l 1", check=False)
    # If tmux is idle (shell prompt) and there's no runner, start one
    if not tmux_is_busy(state):
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

    pick = items[0]
    cloud_id = pick["cloudId"]
    provider = pick["provider"]
    socket = pick.get("socket", "SXM5")
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
        },
        "provider": {"type": provider},
    }
    if ssh_key_id:
        body["pod"]["sshKeyId"] = ssh_key_id

    r = requests.post(f"{API_BASE}/pods/", headers=api_headers(), json=body)
    r.raise_for_status()
    pod = r.json()
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    # Poll until running
    print("Waiting for pod to be ready", end="", flush=True)
    ip = None
    port = 22
    ssh_user = "root"
    for _ in range(120):  # up to 10 minutes
        time.sleep(5)
        print(".", end="", flush=True)
        r = requests.get(
            f"{API_BASE}/pods/status",
            headers=api_headers(),
            params={"pod_ids": pod_id},
        )
        r.raise_for_status()
        statuses = r.json().get("data", [])
        if not statuses:
            continue
        s = statuses[0]
        status = s.get("status", "")
        if status == "ACTIVE":
            ip = s.get("ip")
            ssh_conn = s.get("sshConnection", "")
            # Parse SSH connection string like "ssh root@1.2.3.4 -p 22"
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
            # Check installation status
            inst = s.get("installationStatus")
            if inst in ("FINISHED", None):
                break
            if inst == "ERROR":
                print(f"\nInstallation error: {s.get('installationFailure')}")
                break
        elif status in ("ERROR", "TERMINATED"):
            print(f"\nPod entered {status} state.")
            sys.exit(1)
    print()

    if not ip:
        print("Error: Pod did not become ready in time.")
        sys.exit(1)

    state = {
        "pod_id": pod_id,
        "ip": ip,
        "port": port,
        "ssh_user": ssh_user,
        "name": pod_name,
    }
    # Preserve API key if stored in state
    old_state = load_state()
    if "api_key" in old_state:
        state["api_key"] = old_state["api_key"]
    save_state(state)

    print(f"Pod ready: {ssh_user}@{ip}:{port}")

    # Wait a moment for SSH to be ready
    print("Waiting for SSH...", end="", flush=True)
    for _ in range(30):
        r = remote_exec(state, "echo ok", check=False)
        if r.returncode == 0:
            break
        time.sleep(3)
        print(".", end="", flush=True)
    print()

    # Setup: tmux + repo + uv, then start queue runner
    repo = args.repo
    branch = args.branch
    print("Setting up environment...")
    remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION} || true")
    remote_exec(state, f"touch {QUEUE_FILE}", check=False)

    # Queue the setup commands so they run through the runner
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

    # Start the queue runner in tmux
    runner_escaped = QUEUE_RUNNER.replace("'", "'\\''").replace("\n", " ")
    tmux_send(state, runner_escaped)

    print(f"Queue runner started in tmux session '{TMUX_SESSION}'.")
    print(f"Setup commands queued.")
    print(f"\nSSH: ssh -i {SSH_KEY} -p {port} {ssh_user}@{ip}")
    print(f"Attach: python pi.py ssh")


def cmd_down(args):
    state = require_pod()
    pod_id = state["pod_id"]
    name = state.get("name", pod_id)

    if not args.f:
        # Wait for queued commands to finish
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
    state = require_pod()
    cmd = " ".join(args.command)
    if not cmd:
        print("Error: No command specified.")
        sys.exit(1)

    enqueue_or_flush(state, cmd, force=args.f)


def cmd_train(args):
    state = require_pod()
    config = args.config_path
    overrides = " ".join(args.overrides) if args.overrides else ""

    train_cmd = f"cd ~/nanoGPT && uv run {config} {overrides}".strip()
    enqueue_or_flush(state, train_cmd, force=args.f)

    # Try to infer out_dir from config name
    basename = os.path.basename(config).replace("_config.py", "").replace("_", "-")
    out_dir = f"out-{basename}"
    state["last_out_dir"] = out_dir
    save_state(state)


def cmd_sample(args):
    state = require_pod()
    config = args.config_path
    overrides = " ".join(args.overrides) if args.overrides else ""

    sample_cmd = f"cd ~/nanoGPT && uv run {config} --mode=sample {overrides}".strip()
    enqueue_or_flush(state, sample_cmd, force=args.f)

    # Try to infer out_dir from config name
    basename = os.path.basename(config).replace("_config.py", "").replace("_", "-")
    out_dir = f"out-{basename}"
    state["last_out_dir"] = out_dir
    save_state(state)


def cmd_fetch(args):
    state = require_pod()
    out_dir = args.out_dir or state.get("last_out_dir")
    if not out_dir:
        print("Error: No --out-dir specified and no previous train command to infer from.")
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
        print(f"IP:     {state['ip']}:{state.get('port', 22)}")
        inst = s.get("installationStatus")
        if inst:
            prog = s.get("installationProgress", "")
            print(f"Setup:  {inst}" + (f" ({prog}%)" if prog else ""))
    else:
        print(f"Pod {pod_id}: no status available")
        return

    # Check tmux
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


def cmd_ssh(args):
    state = require_pod()
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
    p_sample.add_argument("config_path", help="Config file path (e.g. config/face_ard_linear_raster_config.py)")
    p_sample.add_argument("overrides", nargs=argparse.REMAINDER, help="Sampling overrides (e.g. --n_step=1)")

    p_fetch = sub.add_parser("fetch", help="Fetch results from the pod")
    p_fetch.add_argument("--images", action="store_true", help="Fetch PNG images")
    p_fetch.add_argument("--checkpoint", action="store_true", help="Fetch checkpoint")
    p_fetch.add_argument("--out-dir", help="Remote output directory name")
    p_fetch.add_argument("--local-dir", default=".", help="Local destination directory")

    sub.add_parser("status", help="Check pod and training status")
    sub.add_parser("ssh", help="SSH into the pod tmux session")
    sub.add_parser("sync", help="Fetch active pods from the API")
    sub.add_parser("list", help="List active pods")

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
        "fetch": cmd_fetch,
        "status": cmd_status,
        "ssh": cmd_ssh,
        "sync": cmd_sync,
        "list": cmd_list,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
