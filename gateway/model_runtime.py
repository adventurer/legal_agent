"""Manage the local vLLM child process started from the web interface."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import psutil

from configs.config import MODEL_PRESETS, VLLM_PORT

ROOT_DIR = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT_DIR / "server" / "vllm_launcher.py"
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._model_key: Optional[str] = None
        self._log_handle = None
        self._started_at: Optional[float] = None
        self._state = "stopped"

    def _api_model(self) -> Optional[str]:
        try:
            response = httpx.get(f"{VLLM_URL}/v1/models", timeout=1.5)
            response.raise_for_status()
            data = response.json().get("data", [])
            return data[0].get("id") if data else None
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            model_name = self._api_model()
            if self._process and self._process.poll() is not None:
                self._state = "stopped"
                self._process = None
                self._model_key = None
                self._started_at = None
                if self._log_handle:
                    self._log_handle.close()
                    self._log_handle = None
            if model_name:
                detected_key = next(
                    (key for key, preset in MODEL_PRESETS.items() if preset["served_name"] == model_name),
                    None,
                )
                if detected_key:
                    # Also reconcile models started outside the page (for example
                    # with `python server/vllm_launcher.py`). The actual /v1/models
                    # response is authoritative for the model name the gateway uses.
                    self._model_key = detected_key
                    self._apply_model(detected_key)
                return {
                    "state": "running",
                    "model": model_name,
                    "model_key": self._model_key or detected_key,
                    "managed": self._process is not None,
                    "started_at": self._started_at,
                }
            return {
                "state": self._state,
                "model": None,
                "model_key": self._model_key,
                "managed": self._process is not None and self._process.poll() is None,
                "started_at": self._started_at,
            }

    def start(self, model_key: str) -> Dict[str, Any]:
        if model_key not in MODEL_PRESETS:
            raise ValueError(f"不支持的模型：{model_key}")
        with self._lock:
            current_name = self._api_model()
            expected_name = MODEL_PRESETS[model_key]["served_name"]
            if current_name and self._process and self._process.poll() is None:
                self._stop_unlocked()
                current_name = None
            if current_name:
                if current_name == expected_name:
                    self._apply_model(model_key)
                    self._model_key = model_key
                    self._state = "running"
                    return self.status_unlocked(current_name)
                self._stop_detected_unlocked(current_name)

            if self._process and self._process.poll() is None:
                if self._model_key == model_key:
                    return {"state": "starting", "model_key": model_key}
                self._stop_unlocked()

            log_path = ROOT_DIR / "data" / "vllm.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(log_path, "a", encoding="utf-8")
            command = [sys.executable, str(LAUNCHER), "--model", model_key, "--no-fp8-kv"]
            kwargs: Dict[str, Any] = {"cwd": str(ROOT_DIR), "stdout": self._log_handle, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            self._process = subprocess.Popen(command, **kwargs)
            self._model_key = model_key
            self._started_at = time.time()
            self._state = "starting"
            return {"state": "starting", "model_key": model_key, "log": str(log_path)}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                model_name = self._api_model()
                if model_name:
                    self._stop_detected_unlocked(model_name)
                self._process = None
                self._model_key = None
                self._state = "stopped"
                return {"state": "stopped"}
            self._stop_unlocked()
            return {"state": "stopped"}

    def _stop_unlocked(self) -> None:
        process = self._process
        if process and process.poll() is None:
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=3)
        self._process = None
        self._model_key = None
        self._started_at = None
        self._state = "stopped"
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def _stop_detected_unlocked(self, model_name: str) -> None:
        """Stop a vLLM process adopted after the gateway itself was restarted."""
        known_models = {preset["served_name"] for preset in MODEL_PRESETS.values()}
        if model_name not in known_models:
            raise RuntimeError(f"端口 {VLLM_PORT} 返回未知模型 {model_name}，为安全起见未停止进程。")

        listener_pids = {
            conn.pid
            for conn in psutil.net_connections(kind="inet")
            if conn.status == psutil.CONN_LISTEN
            and conn.laddr
            and conn.laddr.port == VLLM_PORT
            and conn.pid
        }
        roots = {}
        for pid in listener_pids:
            try:
                process = psutil.Process(pid)
                current = process
                for _ in range(8):
                    command = " ".join(current.cmdline()).lower()
                    if "vllm_launcher.py" in command or "vllm.entrypoints.openai.api_server" in command:
                        roots[current.pid] = current
                        break
                    parent = current.parent()
                    if parent is None or parent.pid == current.pid:
                        break
                    current = parent
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not roots:
            raise RuntimeError(
                f"已检测到模型 {model_name}，但无法确认其 vLLM 启动进程；未终止任何进程。"
            )

        processes = {}
        for root in roots.values():
            try:
                for child in root.children(recursive=True):
                    processes[child.pid] = child
                processes[root.pid] = root
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        targets = list(processes.values())
        for process in reversed(targets):
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=8)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        if alive:
            psutil.wait_procs(alive, timeout=3)

        self._process = None
        self._model_key = None
        self._started_at = None
        self._state = "stopped"

    @staticmethod
    def status_unlocked(model_name: str) -> Dict[str, Any]:
        return {"state": "running", "model": model_name, "model_key": None}

    @staticmethod
    def _apply_model(model_key: str) -> None:
        import sys

        from configs import config
        from gateway import review_orchestrator

        api_server_path = Path(__file__).with_name("api_server.py").resolve()
        main_module = sys.modules.get("__main__")
        if (
            main_module is not None
            and getattr(main_module, "__file__", None)
            and Path(main_module.__file__).resolve() == api_server_path
        ):
            # `python gateway/api_server.py` keeps the live app under __main__.
            # Importing gateway.api_server here would create and mutate a second Agent.
            api_server_module = main_module
        else:
            import gateway.api_server as api_server_module

        agent_instance = api_server_module.agent_instance

        preset = MODEL_PRESETS[model_key]
        agent_instance.model_name = preset["served_name"]
        config.DEFAULT_MODEL_KEY = model_key
        config.DEFAULT_MODEL_NAME = preset["served_name"]
        config.AGENT_CONFIG["model"] = preset["served_name"]
        config.AGENT_CONFIG["max_context_tokens"] = preset["max_model_len"]
        review_orchestrator.MODEL_MAX_CONTEXT = preset["max_model_len"]
        review_orchestrator.CONTEXT_THRESHOLD_95 = int(preset["max_model_len"] * 0.95)


model_runtime = ModelRuntime()
