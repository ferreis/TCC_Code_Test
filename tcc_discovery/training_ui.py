from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from contextual_help import help_for, help_text, help_dataframe

import streamlit as st
import streamlit.components.v1 as components
import torch

from tcc_discovery.runtime_config import (
    configured_stage_ids,
    load_runtime_config,
    materialize_training_plan,
    save_runtime_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "workspace" / "training_frontend"
STATE_FILE = STATE_DIR / "process.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        # Reap completed children so a zombie is not shown as running forever.
        try:
            finished_pid, _ = os.waitpid(int(pid), os.WNOHANG)
            if finished_pid:
                return False
        except ChildProcessError:
            pass
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _gpu_status() -> dict[str, str]:
    if not torch.cuda.is_available():
        return {"GPU": "CPU", "VRAM": "-", "GPU-Util": "-"}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, used, total, util = [x.strip() for x in out.split(",")]
        return {"GPU": name, "VRAM": f"{used}/{total} MiB", "GPU-Util": f"{util}%"}
    except Exception:
        p = torch.cuda.get_device_properties(0)
        return {"GPU": p.name, "VRAM": f"{round(p.total_memory / 1024**3, 1)} GB total", "GPU-Util": "n/d"}


def _tail(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    try:
        # Bound disk reads even for training logs that grow for hours.
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            start = max(0, handle.tell() - 128 * 1024)
            handle.seek(start)
            if start:
                handle.readline()
            text = handle.read().decode("utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        # tqdm rewrites a line with carriage returns.
        content = [line.split("\r")[-1] for line in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(content[-lines:])
    except Exception as exc:
        return f"Erro lendo log: {exc}"


def _stage_dataset_key(stage_id: str) -> str:
    if stage_id in {"det_general", "smoke_det_yolo26n"}:
        return "det_general"
    if stage_id in {"seg_general", "smoke_seg_yolo26n"}:
        return "seg_general"
    if stage_id.startswith("det_fullft_"):
        return stage_id.replace("det_fullft_", "det_lora_")
    if stage_id.startswith("seg_fullft_"):
        return stage_id.replace("seg_fullft_", "seg_lora_")
    return stage_id


def _training_project(runtime: dict[str, Any]) -> Path:
    value = str((runtime.get("paths", {}) or {}).get("training_project", "workspace/training_runs"))
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _render_summary(runtime: dict[str, Any]) -> None:
    summary = _training_project(runtime) / "experiment_summary.csv"
    if summary.exists():
        try:
            df = pd.read_csv(summary)
            if not df.empty:
                st.subheader("Resultados registrados")
                help_dataframe(df.tail(20), use_container_width=True)
        except Exception as exc:
            st.warning(f"Não foi possível ler o resumo: {exc}")


def render_training_page() -> None:
    st.subheader("Treinamento YOLO26", help=help_for('Treinamento YOLO26'))
    help_text("Terminal, Streamlit e Colab usam configs/runtime.yaml + configs/training_experiments.yaml e o mesmo scripts/train_experiments.py.", caption=True)

    runtime_path = REPO_ROOT / "configs" / "runtime.yaml"
    runtime = load_runtime_config(runtime_path)
    profiles = list((runtime.get("profiles", {}) or {}).keys())
    default_profile = str(runtime.get("default_profile", "local"))
    profile = st.selectbox(
        "Perfil de hardware",
        profiles,
        index=profiles.index(default_profile) if default_profile in profiles else 0,
        help=help_for('Perfil de hardware'),
    )

    gpu = _gpu_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("GPU", gpu["GPU"], help=help_for('GPU'))
    c2.metric("VRAM", gpu["VRAM"], help=help_for('VRAM'))
    c3.metric("GPU-Util", gpu["GPU-Util"], help=help_for('GPU-Util'))

    _, plan = materialize_training_plan(
        runtime,
        profile=profile,
        output_path="workspace/frontend_plan_preview.yaml",
        stage_datasets=False,
    )
    stage_ids = configured_stage_ids(plan)
    stage_lookup: dict[str, dict[str, Any]] = {}
    for group in ("smoke_tests", "general_models", "lora_specialists", "full_finetune_controls"):
        stage_lookup.update(plan.get(group, {}) or {})

    stage_id = st.selectbox(
        "Etapa",
        stage_ids,
        format_func=lambda key: f"{key} — {stage_lookup[key].get('label', key)}",
        help=help_for('Etapa'),
    )
    stage = stage_lookup[stage_id]
    dataset_key = _stage_dataset_key(stage_id)

    training = runtime.setdefault("training", {})
    profile_cfg = runtime.setdefault("profiles", {}).setdefault(profile, {})
    smoke = runtime.setdefault("smoke", {})
    lora = runtime.setdefault("lora", {})
    datasets = runtime.setdefault("datasets", {})
    models = runtime.setdefault("models", {})

    with st.form("training_config_form"):
        st.write("Parâmetros compartilhados")
        a, b, c, d = st.columns(4)
        epochs = a.number_input("Epochs", 1, 5000, int(training.get("epochs", 100)), help=help_for('Epochs'))
        sizes = [320, 480, 640, 768, 960, 1024, 1280]
        current_size = int(training.get("imgsz", 640))
        imgsz = b.selectbox("imgsz", sizes, index=sizes.index(current_size) if current_size in sizes else 2, help=help_for('imgsz'))
        batch = c.number_input("Batch", -1, 256, int(profile_cfg.get("batch", 8)), help=help_for('Batch'))
        workers = d.number_input("Workers", 0, 32, int(profile_cfg.get("workers", 2)), help=help_for('Workers'))

        a, b, c, d = st.columns(4)
        device = a.text_input("Device", str(training.get("device", 0)), help=help_for('Device'))
        cache_options = ["false", "disk", "ram"]
        current_cache = str(profile_cfg.get("cache", "disk")).lower()
        cache = b.selectbox("Cache", cache_options, index=cache_options.index(current_cache) if current_cache in cache_options else 1, help=help_for('Cache'))
        amp = c.checkbox("AMP", bool(training.get("amp", True)), help=help_for('AMP'))
        deterministic = d.checkbox("Determinístico", bool(training.get("deterministic", True)), help=help_for('Determinístico'))

        a, b, c, d = st.columns(4)
        patience = a.number_input("Patience", 0, 1000, int(training.get("patience", 30)), help=help_for('Patience'))
        seed = b.number_input("Seed", 0, 2_147_483_647, int(training.get("seed", 42)), help=help_for('Seed'))
        save_period = c.number_input("Salvar a cada N épocas", -1, 1000, int(training.get("save_period", 10)), help=help_for('Salvar a cada N épocas'))
        close_mosaic = d.number_input("Close mosaic", 0, 1000, int(training.get("close_mosaic", 10)), help=help_for('Close mosaic'))

        dataset_path = st.text_input("Dataset data.yaml desta etapa", str(datasets.get(dataset_key, stage.get("data", ""))), help=help_for('Dataset data.yaml desta etapa'))
        if stage.get("task") == "detect" and stage_id in {"det_general", "smoke_det_yolo26n"}:
            model_name = st.text_input("Modelo inicial de detecção", str(models.get("detect", "yolo26n.pt")), help=help_for('Modelo inicial de detecção'))
        elif stage.get("task") == "segment" and stage_id in {"seg_general", "smoke_seg_yolo26n"}:
            model_name = st.text_input("Modelo inicial de segmentação", str(models.get("segment", "yolo26n-seg.pt")), help=help_for('Modelo inicial de segmentação'))
        else:
            model_name = None
            st.info(f"Esta etapa parte do checkpoint geral: {stage.get('base_from')}.")

        if stage_id.startswith("smoke_"):
            help_text("Smoke test")
            smoke_epochs = st.number_input("Smoke epochs", 1, 100, int(smoke.get("epochs", 3)), help=help_for('Smoke epochs'))
            smoke_imgsz = st.number_input("Smoke imgsz", 320, 2048, int(smoke.get("imgsz", 640)), step=32, help=help_for('Smoke imgsz'))
        else:
            smoke_epochs, smoke_imgsz = int(smoke.get("epochs", 3)), int(smoke.get("imgsz", 640))

        if str(stage.get("regime", "")).startswith("lora"):
            help_text("LoRA")
            a, b, c, d = st.columns(4)
            lora_r = a.number_input("Rank (r)", 1, 256, int(lora.get("r", 16)), help=help_for('Rank (r)'))
            lora_alpha = b.number_input("Alpha", 1, 1024, int(lora.get("alpha", 32)), help=help_for('Alpha'))
            lora_dropout = c.number_input("Dropout", 0.0, 0.9, float(lora.get("dropout", 0.10)), 0.01, help=help_for('Dropout'))
            lora_head = d.checkbox("Incluir head", bool(lora.get("include_head", True)), help=help_for('Incluir head'))
            lora_gc = st.checkbox("Gradient checkpointing", bool(lora.get("gradient_checkpointing", True)), help=help_for('Gradient checkpointing'))
        else:
            lora_r, lora_alpha = int(lora.get("r", 16)), int(lora.get("alpha", 32))
            lora_dropout = float(lora.get("dropout", 0.10))
            lora_head = bool(lora.get("include_head", True))
            lora_gc = bool(lora.get("gradient_checkpointing", True))

        save_cfg = st.form_submit_button("Salvar parâmetros", type="secondary")
        if save_cfg:
            training.update(
                {
                    "epochs": int(epochs),
                    "imgsz": int(imgsz),
                    "device": int(device) if str(device).lstrip("-").isdigit() else str(device),
                    "amp": bool(amp),
                    "patience": int(patience),
                    "seed": int(seed),
                    "deterministic": bool(deterministic),
                    "save_period": int(save_period),
                    "close_mosaic": int(close_mosaic),
                }
            )
            profile_cfg.update(
                {
                    "batch": int(batch),
                    "workers": int(workers),
                    "cache": False if cache == "false" else cache,
                }
            )
            datasets[dataset_key] = dataset_path
            if model_name is not None:
                models["detect" if stage.get("task") == "detect" else "segment"] = model_name
            smoke.update({"epochs": int(smoke_epochs), "imgsz": int(smoke_imgsz)})
            lora.update(
                {
                    "r": int(lora_r),
                    "alpha": int(lora_alpha),
                    "dropout": float(lora_dropout),
                    "gradient_checkpointing": bool(lora_gc),
                    "include_head": bool(lora_head),
                }
            )
            runtime["default_profile"] = profile
            save_runtime_config(runtime, runtime_path)
            st.success("configs/runtime.yaml atualizado.")

    lora_available = importlib.util.find_spec("ultralytics.utils.lora") is not None
    if str(stage.get("regime", "")).startswith("lora") and not lora_available:
        st.warning("YOLO-Master/LoRA não está disponível neste ambiente.")
        if st.button("Instalar/atualizar YOLO-Master neste ambiente", help=help_for('Instalar/atualizar YOLO-Master neste ambiente')):
            yolo_dir = Path(str(profile_cfg.get("yolo_master_dir", ".yolo-master"))).expanduser()
            if not yolo_dir.is_absolute():
                yolo_dir = (REPO_ROOT / yolo_dir).resolve()
            result = subprocess.run(
                [sys.executable, "scripts/setup_yolo_master.py", "--dir", str(yolo_dir)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            st.code(result.stdout, language="text")
            if result.returncode == 0:
                st.success("Backend instalado. O treinamento roda em um processo Python novo e já poderá usar a LoRA.")
            else:
                st.error("Falha ao instalar YOLO-Master.")

    _render_execution(stage_id, profile, runtime_path, runtime)


@st.fragment(run_every="1s")
def _render_execution(
    stage_id: str, profile: str, runtime_path: Path, runtime: dict[str, Any]
) -> None:
    state = _load_state()
    running = _pid_alive(state.get("pid"))
    st.divider()
    st.write("Execução")
    help_text("Salve os parâmetros acima antes de iniciar. O treino é executado em segundo plano e continua mesmo com reruns da página.", caption=True)
    b1, b2 = st.columns(2)

    if b1.button("Iniciar treinamento", type="primary", disabled=running, help=help_for('Iniciar treinamento')):
        log_path = STATE_DIR / f"{stage_id}.log"
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        cmd = [
            sys.executable,
            "-u",
            "scripts/train_experiments.py",
            "--runtime-config",
            str(runtime_path),
            "--profile",
            profile,
            "--only",
            stage_id,
        ]
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_handle.close()
        state = {
            "pid": process.pid,
            "stage_id": stage_id,
            "profile": profile,
            "log": str(log_path),
            "started_at_utc": _utc_now(),
            "command": cmd,
        }
        _save_state(state)
        st.success(f"Treinamento iniciado. PID {process.pid}")
        st.rerun()

    if b2.button("Parar treinamento", disabled=not running, help=help_for('Parar treinamento')):
        try:
            os.killpg(os.getpgid(int(state["pid"])), signal.SIGTERM)
            st.warning("Sinal de parada enviado ao treinamento.")
        except Exception as exc:
            st.error(f"Não foi possível parar: {exc}")
        st.rerun()

    state = _load_state()
    running = _pid_alive(state.get("pid"))
    if state:
        st.write(
            f"**Estado:** {'executando' if running else 'encerrado'} | "
            f"**Etapa:** {state.get('stage_id')} | **PID:** {state.get('pid')} | "
            f"**Início:** {state.get('started_at_utc')}"
        )
        log_path = Path(str(state.get("log", "")))
        log_text = _tail(log_path)
        st.caption("Log ao vivo • atualização automática a cada segundo")
        with st.popover("ⓘ Termos do log"):
            help_text("Epochs · batch · learning rate · loss · precision · recall · mAP50 · mAP50-95 · checkpoint · VRAM · ETA")
        _render_terminal(log_text or "Aguardando saída do treinamento…")

    _render_summary(runtime)


def _render_terminal(log_text: str) -> None:
    # Escape process output before embedding it in the terminal document.
    components.html(
        """<!doctype html><html><head><style>
        html, body { margin: 0; height: 100%; background: #0e1117; }
        #terminal { box-sizing: border-box; height: 400px; overflow: auto;
          padding: 16px; border: 1px solid #30363d; border-radius: 8px;
          color: #e6edf3; font: 12px/1.5 ui-monospace, monospace;
          white-space: pre-wrap; overflow-wrap: anywhere; }
        </style></head><body><div id="terminal" role="log"
        aria-label="Log do treinamento" tabindex="0">"""
        + html.escape(log_text)
        + """</div><script>
        const terminal = document.getElementById('terminal');
        terminal.scrollTop = terminal.scrollHeight;
        </script></body></html>""",
        height=400,
        scrolling=False,
    )
