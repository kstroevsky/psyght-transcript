"""Tests for Dockerized Ollama/Qwen bootstrap."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.tools import setup_ollama_qwen as cli


class SetupOllamaQwenTest(unittest.TestCase):
    def test_ensure_container_creates_container_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(
                cli,
                "_docker_run",
                side_effect=[
                    subprocess.CompletedProcess(["docker"], 1, "", "missing"),
                    subprocess.CompletedProcess(["docker"], 0, "container-id\n", ""),
                ],
            ) as docker_run_mock:
                previous_status = cli.ensure_container(
                    docker_bin="docker",
                    image="ollama/ollama:latest",
                    container_name="phase1-ollama-qwen",
                    host_port=11434,
                    volume_dir=tmp_dir,
                )

        self.assertEqual("created", previous_status)
        run_command = docker_run_mock.call_args_list[1].args[1:]
        self.assertEqual("run", run_command[0])
        self.assertIn("phase1-ollama-qwen", run_command)

    def test_ensure_container_starts_existing_stopped_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(
                cli,
                "_docker_run",
                side_effect=[
                    subprocess.CompletedProcess(["docker"], 0, "exited\n", ""),
                    subprocess.CompletedProcess(["docker"], 0, "phase1-ollama-qwen\n", ""),
                ],
            ) as docker_run_mock:
                previous_status = cli.ensure_container(
                    docker_bin="docker",
                    image="ollama/ollama:latest",
                    container_name="phase1-ollama-qwen",
                    host_port=11434,
                    volume_dir=tmp_dir,
                )

        self.assertEqual("exited", previous_status)
        start_command = docker_run_mock.call_args_list[1].args[1:]
        self.assertEqual(("start", "phase1-ollama-qwen"), start_command)

    def test_ensure_container_reuses_running_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(
                cli,
                "_docker_run",
                return_value=subprocess.CompletedProcess(["docker"], 0, "running\n", ""),
            ) as docker_run_mock:
                previous_status = cli.ensure_container(
                    docker_bin="docker",
                    image="ollama/ollama:latest",
                    container_name="phase1-ollama-qwen",
                    host_port=11434,
                    volume_dir=tmp_dir,
                )

        self.assertEqual("running", previous_status)
        self.assertEqual(1, docker_run_mock.call_count)

    def test_pull_model_invokes_qwen_pull(self) -> None:
        with patch.object(cli, "_docker_run") as docker_run_mock:
            cli.pull_model(docker_bin="docker", container_name="phase1-ollama-qwen", model="qwen3:8b")

        self.assertEqual(
            ("docker", "exec", "phase1-ollama-qwen", "ollama", "pull", "qwen3:8b"),
            docker_run_mock.call_args.args,
        )

    def test_wait_until_ready_times_out_with_clear_error(self) -> None:
        with (
            patch.object(cli, "_request_json", side_effect=RuntimeError("connection refused")),
            patch.object(cli.time, "sleep"),
            patch.object(cli.time, "time", side_effect=[0.0, 0.1, 0.2, 0.3]),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                cli.wait_until_ready("http://127.0.0.1:11434", timeout_sec=0.25, poll_interval_sec=0.01)

    def test_request_json_wraps_socket_errors(self) -> None:
        with patch.object(cli, "urlopen", side_effect=ConnectionResetError("reset by peer")):
            with self.assertRaisesRegex(RuntimeError, "reset by peer"):
                cli._request_json("http://127.0.0.1:11434/api/tags")

    def test_main_returns_nonzero_when_verification_fails(self) -> None:
        with (
            patch.object(cli, "run_setup", side_effect=RuntimeError("empty response")),
            patch.object(sys, "argv", ["setup_ollama_qwen.py"]),
        ):
            result = cli.main()

        self.assertEqual(1, result)

    def test_verify_model_uses_non_thinking_short_generation(self) -> None:
        with patch.object(cli, "_request_json", return_value={"response": "готово"}) as request_mock:
            response = cli.verify_model(
                base_url="http://127.0.0.1:11434",
                model="qwen3:8b",
                prompt="Respond with one short word in Russian.",
            )

        self.assertEqual("готово", response)
        payload = request_mock.call_args.args[1]
        self.assertEqual("qwen3:8b", payload["model"])
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(0, payload["options"]["temperature"])
        self.assertEqual(1, payload["options"]["top_k"])
        self.assertEqual(16, payload["options"]["num_predict"])


if __name__ == "__main__":
    unittest.main()
