"""Bootstrap a reusable Dockerized Ollama instance with qwen3:8b loaded."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _phase1_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_volume_dir() -> Path:
    return _phase1_dir() / ".ollama"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start Dockerized Ollama and preload qwen3:8b.")
    parser.add_argument("--docker-bin", default=shutil.which("docker") or "docker")
    parser.add_argument("--image", default="ollama/ollama:latest")
    parser.add_argument("--container-name", default="phase1-ollama-qwen")
    parser.add_argument("--host-port", type=int, default=11434)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--volume-dir", default=str(_default_volume_dir()))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ready-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--verify-prompt", default="Respond with one short word in Russian.")
    return parser


def _docker_run(
    docker_bin: str,
    *args: str,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [docker_bin, *args],
        check=check,
        capture_output=capture_output,
        text=text,
    )


def _require_docker(docker_bin: str) -> None:
    if shutil.which(docker_bin) is None:
        raise RuntimeError(f"Docker CLI not found: {docker_bin}")
    try:
        _docker_run(docker_bin, "version")
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(
            f"Docker is unavailable: {details}. Start Docker Desktop (or another Docker daemon) "
            "and wait until `docker version` succeeds."
        ) from exc


def _container_status(docker_bin: str, container_name: str) -> str | None:
    completed = _docker_run(
        docker_bin,
        "container",
        "inspect",
        container_name,
        "--format",
        "{{.State.Status}}",
        check=False,
    )
    if completed.returncode != 0:
        return None
    status = completed.stdout.strip()
    return status or None


def ensure_container(
    *,
    docker_bin: str,
    image: str,
    container_name: str,
    host_port: int,
    volume_dir: str | Path,
) -> str:
    volume_path = Path(volume_dir).expanduser().resolve()
    volume_path.mkdir(parents=True, exist_ok=True)
    status = _container_status(docker_bin, container_name)
    if status == "running":
        print(f"[OLLAMA] Reusing running container {container_name}")
        return "running"
    if status is not None:
        _docker_run(docker_bin, "start", container_name)
        print(f"[OLLAMA] Started existing container {container_name} (was {status})")
        return status

    _docker_run(
        docker_bin,
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{host_port}:11434",
        "-v",
        f"{volume_path}:/root/.ollama",
        image,
    )
    print(f"[OLLAMA] Created container {container_name} from {image}")
    return "created"


def _request_json(url: str, payload: dict[str, object] | None = None, *, timeout_sec: float = 30.0) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def wait_until_ready(base_url: str, *, timeout_sec: float, poll_interval_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = "service did not become ready"
    while time.time() < deadline:
        try:
            _request_json(f"{base_url.rstrip('/')}/api/tags", timeout_sec=min(10.0, poll_interval_sec + 5.0))
            print(f"[OLLAMA] Ready at {base_url}")
            return
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(poll_interval_sec)
    raise RuntimeError(f"Ollama readiness check timed out after {timeout_sec:.1f}s: {last_error}")


def pull_model(*, docker_bin: str, container_name: str, model: str) -> None:
    print(f"[OLLAMA] Pulling model {model}")
    _docker_run(docker_bin, "exec", container_name, "ollama", "pull", model, capture_output=False)


def verify_model(*, base_url: str, model: str, prompt: str) -> str:
    payload = _request_json(
        f"{base_url.rstrip('/')}/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "top_k": 1,
                "num_predict": 16,
            },
        },
        timeout_sec=120.0,
    )
    response = str(payload.get("response") or "").strip()
    if not response:
        raise RuntimeError(f"Ollama verification returned an empty response for model {model}.")
    print(f"[OLLAMA] Verification response: {response}")
    return response


def run_setup(args: argparse.Namespace) -> None:
    _require_docker(args.docker_bin)
    ensure_container(
        docker_bin=args.docker_bin,
        image=args.image,
        container_name=args.container_name,
        host_port=args.host_port,
        volume_dir=args.volume_dir,
    )
    wait_until_ready(
        args.base_url,
        timeout_sec=float(args.ready_timeout_sec),
        poll_interval_sec=float(args.poll_interval_sec),
    )
    pull_model(
        docker_bin=args.docker_bin,
        container_name=args.container_name,
        model=args.model,
    )
    verify_model(base_url=args.base_url, model=args.model, prompt=args.verify_prompt)
    print(f"[OLLAMA] base_url={args.base_url}")
    print(f"[OLLAMA] container_name={args.container_name}")
    print(f"[OLLAMA] model={args.model}")
    print("[OLLAMA] status=ready")


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_setup(args)
    except Exception as exc:
        print(f"[OLLAMA] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
