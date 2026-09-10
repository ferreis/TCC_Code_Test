from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_GROUPS = ("smoke_tests", "general_models", "lora_specialists", "full_finetune_controls")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_runtime_config(path: str | Path = "configs/runtime.yaml") -> dict[str, Any]:
    config_path = _resolve_repo_path(path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuração runtime inválida: {config_path}")
    cfg["_config_path"] = str(config_path)
    return cfg


def save_runtime_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    payload = copy.deepcopy(config)
    config_path = _resolve_repo_path(path or payload.pop("_config_path", "configs/runtime.yaml"))
    payload.pop("_config_path", None)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def runtime_profile(config: dict[str, Any], profile: str | None = None) -> tuple[str, dict[str, Any]]:
    profile_name = profile or str(config.get("default_profile", "local"))
    profiles = config.get("profiles", {}) or {}
    if profile_name not in profiles:
        raise KeyError(f"Perfil runtime não encontrado: {profile_name}")
    merged = dict(config.get("training", {}) or {})
    merged.update(profiles[profile_name] or {})
    return profile_name, merged


def configured_stage_ids(plan: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for group in TRAIN_GROUPS:
        ids.extend((plan.get(group, {}) or {}).keys())
    return ids


def _dataset_for_stage(runtime: dict[str, Any], stage_id: str) -> str | None:
    datasets = runtime.get("datasets", {}) or {}
    if stage_id in {"smoke_det_yolo26n", "det_general"}:
        return datasets.get("det_general")
    if stage_id in {"smoke_seg_yolo26n", "seg_general"}:
        return datasets.get("seg_general")
    if stage_id.startswith("det_fullft_"):
        return datasets.get(stage_id.replace("det_fullft_", "det_lora_"))
    if stage_id.startswith("seg_fullft_"):
        return datasets.get(stage_id.replace("seg_fullft_", "seg_lora_"))
    return datasets.get(stage_id)


def _stage_dataset(data_yaml: str, stage_id: str, profile_cfg: dict[str, Any]) -> str:
    if not bool(profile_cfg.get("use_local_dataset_cache", False)):
        return str(_resolve_repo_path(data_yaml))

    source_yaml = _resolve_repo_path(data_yaml)
    if not source_yaml.exists():
        return str(source_yaml)

    source_root = source_yaml.parent
    local_root = Path(str(profile_cfg.get("local_data_root", "workspace/local_training_data"))).expanduser()
    if not local_root.is_absolute():
        local_root = (REPO_ROOT / local_root).resolve()
    target_root = local_root / stage_id
    refresh = bool(profile_cfg.get("refresh_local_dataset_cache", False))

    if refresh and target_root.exists():
        shutil.rmtree(target_root)
    if not target_root.exists():
        target_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, target_root)

    target_yaml = target_root / source_yaml.name
    if not target_yaml.exists():
        raise FileNotFoundError(f"Falha no staging do dataset: {target_yaml}")
    return str(target_yaml)


def materialize_training_plan(
    runtime_config: dict[str, Any],
    *,
    profile: str | None = None,
    base_plan_path: str | Path | None = None,
    output_path: str | Path = "workspace/training_plan_runtime.yaml",
    selected_stages: Iterable[str] | None = None,
    stage_datasets: bool = False,
) -> tuple[Path, dict[str, Any]]:
    profile_name, profile_cfg = runtime_profile(runtime_config, profile)
    paths = runtime_config.get("paths", {}) or {}
    plan_source = base_plan_path or paths.get("training_plan", "configs/training_experiments.yaml")
    plan_path = _resolve_repo_path(plan_source)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(plan, dict):
        raise ValueError(f"Plano de treinamento inválido: {plan_path}")

    training = dict(runtime_config.get("training", {}) or {})
    training.update(profile_cfg)
    allowed_global = (
        "epochs", "imgsz", "batch", "device", "workers", "amp", "optimizer",
        "patience", "seed", "deterministic", "cache", "close_mosaic", "save_period",
        "plots", "verbose",
    )
    global_cfg = {key: training[key] for key in allowed_global if key in training}
    global_cfg["project"] = str(paths.get("training_project", "workspace/training_runs"))
    global_cfg["eval_split"] = str(training.get("eval_split", "test"))
    plan["global"] = global_cfg

    models = runtime_config.get("models", {}) or {}
    smoke = runtime_config.get("smoke", {}) or {}
    lora = runtime_config.get("lora", {}) or {}
    explicit_selection = selected_stages is not None
    chosen = set(selected_stages or configured_stage_ids(plan))

    for group in TRAIN_GROUPS:
        for stage_id, item in (plan.get(group, {}) or {}).items():
            if explicit_selection:
                # --only deve executar a etapa escolhida mesmo que o template versionado
                # esteja enabled:false para evitar disparos acidentais em lote.
                item["enabled"] = stage_id in chosen

            if stage_id in {"smoke_det_yolo26n", "det_general"}:
                item["model"] = str(models.get("detect", "yolo26n.pt"))
            elif stage_id in {"smoke_seg_yolo26n", "seg_general"}:
                item["model"] = str(models.get("segment", "yolo26n-seg.pt"))

            dataset = _dataset_for_stage(runtime_config, stage_id)
            if dataset:
                item["data"] = (
                    _stage_dataset(str(dataset), stage_id, profile_cfg)
                    if stage_datasets and stage_id in chosen
                    else str(_resolve_repo_path(str(dataset)))
                )

            train_args = dict(item.get("train_args", {}) or {})
            if stage_id.startswith("smoke_"):
                for key in ("epochs", "imgsz", "batch", "workers", "cache"):
                    if key in smoke:
                        train_args[key] = smoke[key]
                    elif key in profile_cfg:
                        train_args[key] = profile_cfg[key]
            if str(item.get("regime", "")).startswith("lora"):
                train_args.update(
                    {
                        "lora_r": int(lora.get("r", 16)),
                        "lora_alpha": int(lora.get("alpha", 32)),
                        "lora_dropout": float(lora.get("dropout", 0.10)),
                        "lora_gradient_checkpointing": bool(lora.get("gradient_checkpointing", True)),
                        "lora_include_head": bool(lora.get("include_head", True)),
                    }
                )
            item["train_args"] = train_args

    plan["runtime"] = {
        "profile": profile_name,
        "runtime_config": runtime_config.get("_config_path"),
    }

    destination = _resolve_repo_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination, plan
