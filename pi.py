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
    key = os.environ.get("PRIME_API_KEY")
    if not key:
        state = load_state()
        key = state.get("api_key")
    if not key:
        print("Error: Set PRIME_API_KEY env var or store api_key in ~/.pi_state.json")
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
    """Check if the tmux pane is running something (not just a shell)."""
    r = remote_exec(state, f"tmux list-panes -t {TMUX_SESSION} -F '#{{pane_current_command}}'", check=False)
    if r.returncode != 0:
        return False
    cmd = r.stdout.strip()
    return cmd not in ("bash", "zsh", "sh", "fish", "")


def tmux_capture(state, lines=5):
    """Capture recent tmux output."""
    r = remote_exec(state, f"tmux capture-pane -t {TMUX_SESSION} -p -l {lines}", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


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

    # Setup: tmux + repo + uv
    repo = args.repo
    branch = args.branch
    print("Setting up environment...")
    remote_exec(state, f"tmux new-session -d -s {TMUX_SESSION} || true")
    setup_cmds = " && ".join([
        f"git clone --depth 1 --branch {branch} {repo}",
        "cd nanoGPT",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        'export PATH="$HOME/.local/bin:$PATH"',
        "uv python install 3.13",
        "uv venv --python 3.13",
        "uv sync",
    ])
    tmux_send(state, setup_cmds)
    print(f"Setup commands sent to tmux session '{TMUX_SESSION}'.")
    print(f"\nSSH: ssh -i {SSH_KEY} -p {port} {ssh_user}@{ip}")
    print(f"Attach: python pi.py ssh")


def cmd_down(args):
    state = require_pod()
    pod_id = state["pod_id"]
    name = state.get("name", pod_id)
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

    if tmux_is_busy(state):
        print("Warning: A command is already running in the tmux session.")
        resp = input("Send anyway? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    tmux_send(state, cmd)
    print(f"Sent: {cmd}")


def cmd_train(args):
    state = require_pod()
    config = args.config_path
    overrides = " ".join(args.overrides) if args.overrides else ""

    train_cmd = f"cd ~/nanoGPT && uv run {config} {overrides}".strip()
    sample_cmd = f"cd ~/nanoGPT && uv run {config} --mode=sample {overrides}".strip()
    full_cmd = f"{train_cmd} && {sample_cmd}"

    if tmux_is_busy(state):
        print("Warning: A command is already running in the tmux session.")
        resp = input("Send anyway? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    tmux_send(state, full_cmd)
    print(f"Training: {train_cmd}")
    print(f"Then sampling: {sample_cmd}")

    # Try to infer out_dir from config name
    # e.g. config/face_ard_linear_raster_config.py -> out-face-ard-linear-raster
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

    output = tmux_capture(state, lines=8)
    if output:
        print(f"\nRecent output:\n{output}")


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

    sub.add_parser("down", help="Terminate the current pod")

    p_run = sub.add_parser("run", help="Run a command on the remote pod")
    p_run.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    p_train = sub.add_parser("train", help="Train then sample")
    p_train.add_argument("config_path", help="Config file path (e.g. config/face_ard_linear_raster_config.py)")
    p_train.add_argument("overrides", nargs="*", help="Training overrides (e.g. --n_step=1)")

    p_fetch = sub.add_parser("fetch", help="Fetch results from the pod")
    p_fetch.add_argument("--images", action="store_true", help="Fetch PNG images")
    p_fetch.add_argument("--checkpoint", action="store_true", help="Fetch checkpoint")
    p_fetch.add_argument("--out-dir", help="Remote output directory name")
    p_fetch.add_argument("--local-dir", default=".", help="Local destination directory")

    sub.add_parser("status", help="Check pod and training status")
    sub.add_parser("ssh", help="SSH into the pod tmux session")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "up": cmd_up,
        "down": cmd_down,
        "run": cmd_run,
        "train": cmd_train,
        "fetch": cmd_fetch,
        "status": cmd_status,
        "ssh": cmd_ssh,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
