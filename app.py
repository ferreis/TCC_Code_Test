from __future__ import annotations

from collections import Counter
from pathlib import Path

import streamlit as st

from tcc_discovery.augment import augment_project
from tcc_discovery.box_clustering import (
    BoxClusteringConfig,
    export_bbox_manifest,
    render_box_overlay,
    run_box_clustering,
)
from tcc_discovery.core import (
    DiscoveryConfig,
    export_yolo,
    load_project,
    render_object_overlay,
    run_discovery,
    save_project,
)
from tcc_discovery.upload_clustering import prepare_uploaded_media

APP_VERSION = "0.5-upload-clustering"
_KEEP = object()
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _render_overlay(project_dir: Path, obj: dict):
    if obj.get("mask_path"):
        return render_object_overlay(project_dir, obj)
    return render_box_overlay(obj)


def _update_objects(project_dir: Path, object_ids, label=_KEEP, ignored=_KEEP) -> int:
    ids = set(object_ids)
    catalog = load_project(project_dir)
    count = 0
    for obj in catalog.get("objects", []):
        if obj.get("id") not in ids:
            continue
        if label is not _KEEP:
            obj["label"] = label.strip() if isinstance(label, str) and label.strip() else None
        if ignored is not _KEEP:
            obj["ignored"] = bool(ignored)
        count += 1
    save_project(project_dir, catalog)
    return count


def _show_thumbnails(project_dir: Path, group: list[dict], key: str) -> None:
    max_thumbs = min(60, len(group))
    if max_thumbs <= 0:
        return
    if max_thumbs <= 4:
        show_n = max_thumbs
        st.caption(f"Mostrando todas as {show_n} miniaturas deste grupo.")
    else:
        show_n = st.slider(
            "Quantidade de miniaturas",
            min_value=4,
            max_value=max_thumbs,
            value=min(20, max_thumbs),
            key=f"thumbs_{key}",
        )
    cols = st.columns(min(5, show_n))
    for i, obj in enumerate(group[:show_n]):
        crop = project_dir / obj["crop_path"]
        with cols[i % len(cols)]:
            caption = f"{obj['id']}\ncluster={obj.get('cluster')}\nlabel={obj.get('label')}"
            st.image(str(crop), caption=caption, use_container_width=True)


def _project_or_stop(project_dir: Path) -> dict:
    if not (project_dir / "catalog.json").exists():
        st.warning("Projeto não encontrado. Execute uma descoberta, upload ou clustering pelas bounding boxes primeiro.")
        st.stop()
    catalog = load_project(project_dir)
    if not catalog.get("objects"):
        st.warning("O projeto não possui objetos.")
        st.stop()
    return catalog


def _progress_widgets():
    bar = st.progress(0.0)
    status = st.empty()

    def progress(done, total, message):
        if total:
            bar.progress(min(1.0, max(0.0, done / total)))
        status.write(message)

    return bar, status, progress


st.set_page_config(page_title="TCC - Object Discovery Lab", layout="wide")
st.title("Object Discovery Lab")
st.caption(
    f"{APP_VERSION} · upload de imagem/vídeo, descoberta SAM, clustering pelas boxes e revisão humana."
)

page = st.sidebar.radio(
    "Etapa",
    [
        "0. Upload e clusterizar",
        "1. Descobrir com SAM",
        "2. Clusterizar bounding boxes",
        "3. Classificar pendentes",
        "4. Revisar classificados",
        "5. Revisar ignorados",
        "6. Exportar",
        "7. Aumentar dataset",
    ],
)
project_default = "workspace/upload_cluster" if page == "0. Upload e clusterizar" else "workspace/discovery"
project_dir = Path(st.sidebar.text_input("Projeto local", project_default))


if page == "0. Upload e clusterizar":
    st.subheader("Upload rápido de imagem ou vídeo")
    st.write(
        "Envie uma imagem ou vídeo. Imagens são processadas diretamente; vídeos são amostrados em frames. "
        "Em seguida o SAM gera candidatos, o DINOv2 extrai embeddings e o HDBSCAN forma os clusters."
    )
    uploaded = st.file_uploader(
        "Imagem ou vídeo",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "mp4", "mov", "avi", "mkv", "m4v", "webm"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        is_video = suffix in VIDEO_SUFFIXES
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.caption(f"Arquivo: {uploaded.name} · {size_mb:.1f} MB")
        if is_video:
            st.video(uploaded)
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if is_video:
                sample_every = st.selectbox(
                    "Extrair 1 frame a cada",
                    [0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
                    index=2,
                    format_func=lambda x: f"{x:g} s",
                )
                max_frames = st.number_input("Máximo de frames", 1, 2000, 120, step=10)
            else:
                sample_every = 1.0
                max_frames = 1
            checkpoint = st.text_input("Checkpoint SAM", "weights/sam_vit_b_01ec64.pth", key="upload_sam_checkpoint")

        with c2:
            points = st.select_slider("SAM points_per_side", [8, 12, 16, 24, 32], value=16, key="upload_points")
            points_batch = st.selectbox("SAM points_per_batch", [1, 2, 4, 8, 16], index=2, key="upload_points_batch")
            max_objects = st.number_input("Máx. objetos por frame", 1, 300, 30, key="upload_max_objects")
            min_area = st.number_input("Área mínima da máscara (px)", 10, 100000, 80, key="upload_min_area")

        with c3:
            auto_tile_size = st.selectbox("Auto-tile", [256, 384, 512, 640, 768, 1024], index=2, key="upload_tile")
            min_cluster = st.number_input("Mínimo por cluster", 2, 100, 3, key="upload_min_cluster")
            embedding_batch = st.selectbox("Batch DINOv2", [1, 2, 4, 8, 16, 32], index=2, key="upload_dino_batch")
            overwrite = st.checkbox("Recriar projeto", value=False, key="upload_overwrite")

        st.info(
            "Para a GTX 1660, os valores padrão são conservadores. No Colab com GPU mais forte você pode aumentar "
            "points_per_side e o batch do DINOv2. O vídeo mantém referência ao número do frame original."
        )

        if st.button("Preparar upload e clusterizar", type="primary"):
            try:
                prep_bar, prep_status, prep_progress = _progress_widgets()
                prep_status.write("Preparando mídia...")
                media_info = prepare_uploaded_media(
                    uploaded.getvalue(),
                    uploaded.name,
                    sample_every_seconds=float(sample_every),
                    max_frames=int(max_frames),
                    overwrite=True,
                    progress_cb=prep_progress,
                )
                prep_bar.progress(1.0)

                cfg = DiscoveryConfig(
                    sam_checkpoint=checkpoint,
                    points_per_side=int(points),
                    points_per_batch=int(points_batch),
                    crop_n_layers=0,
                    min_area_px=int(min_area),
                    max_objects_per_image=int(max_objects),
                    tile_size=0,
                    auto_tile_large_images=True,
                    auto_tile_size=int(auto_tile_size),
                    auto_tile_trigger_px=1024,
                    embedding_batch_size=int(embedding_batch),
                    min_cluster_size=int(min_cluster),
                    cpu_fallback_on_oom=True,
                    dino_cpu_fallback_on_oom=True,
                )
                bar, status, progress = _progress_widgets()
                catalog = run_discovery(
                    media_info["frames_dir"],
                    project_dir,
                    cfg,
                    overwrite=overwrite,
                    progress_cb=progress,
                )
                catalog["project_type"] = "upload_clustering"
                catalog["upload_source"] = media_info
                save_project(project_dir, catalog)
                bar.progress(1.0)
                status.write("Clustering concluído")

                counts = Counter(int(o.get("cluster", -1)) for o in catalog.get("objects", []))
                regular_clusters = [c for c in counts if c != -1]
                m1, m2, m3 = st.columns(3)
                m1.metric("Frames processados", len(catalog.get("images", [])))
                m2.metric("Objetos propostos", len(catalog.get("objects", [])))
                m3.metric("Clusters", len(regular_clusters))
                st.success("Upload processado e clusterizado. Use '3. Classificar pendentes' para revisar os grupos.")
                if counts:
                    st.write("Distribuição dos maiores clusters:")
                    st.json({str(k): int(v) for k, v in counts.most_common(12)})
            except Exception as exc:
                st.exception(exc)
    else:
        st.caption("Formatos aceitos: JPG/PNG/TIFF/WebP e MP4/MOV/AVI/MKV/M4V/WebM.")


elif page == "1. Descobrir com SAM":
    st.subheader("Descoberta não supervisionada com SAM")
    st.caption(
        "Esta etapa é o baseline totalmente automático. Os tiles são internos e as coordenadas voltam para a imagem original, "
        "mas um objeto cortado exatamente na borda de tiles ainda pode gerar fragmentos separados."
    )
    input_dir = st.text_input("Pasta de imagens", "data/roboflow/train/images")
    checkpoint = st.text_input("Checkpoint SAM", "weights/sam_vit_b_01ec64.pth")

    low_memory = st.checkbox(
        "Modo seguro para GPU de 6 GB",
        value=True,
        help="Usa batches pequenos, auto-tiling e fallback para CPU se a VRAM acabar.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        points = st.select_slider(
            "SAM points_per_side", [8, 12, 16, 24, 32, 48, 64], value=16 if low_memory else 32
        )
        points_batch = st.selectbox(
            "SAM points_per_batch", [1, 2, 4, 8, 16, 32, 64], index=2 if low_memory else 4
        )
        crop_layers = st.selectbox("SAM crop_n_layers", [0, 1, 2], index=0 if low_memory else 1)
        min_area = st.number_input("Área mínima da máscara (px)", 10, 100000, 120)

    with c2:
        max_objects = st.number_input("Máx. objetos/imagem", 1, 300, 30 if low_memory else 50)
        manual_tile = st.selectbox(
            "Tile manual (0 = automático)", [0, 256, 384, 512, 640, 768, 1024, 1280, 1536], index=0
        )
        auto_tile = st.checkbox("Auto-tile para imagens grandes", value=True)
        auto_tile_size = st.selectbox(
            "Tamanho do tile automático", [256, 384, 512, 640, 768, 1024], index=2 if low_memory else 4
        )
        auto_tile_trigger = st.selectbox(
            "Ativar auto-tile acima de", [768, 1024, 1280, 1536, 2048, 3072], index=1 if low_memory else 3
        )

    with c3:
        min_cluster = st.number_input("Mínimo por cluster", 2, 100, 3 if low_memory else 6)
        embedding_batch = st.selectbox("Batch DINOv2", [1, 2, 4, 8, 16, 32], index=2 if low_memory else 3)
        cpu_fallback = st.checkbox("Fallback SAM para CPU se faltar VRAM", value=True)
        dino_cpu_fallback = st.checkbox("Fallback DINOv2 para CPU", value=True)
        overwrite = st.checkbox("Recriar projeto", value=False)

    if low_memory:
        st.info(
            "Perfil GTX 1660: points_per_side=16, points_per_batch=4, crop_n_layers=0, auto-tile=512 e DINO batch=4."
        )

    if st.button("Executar descoberta", type="primary"):
        cfg = DiscoveryConfig(
            sam_checkpoint=checkpoint,
            points_per_side=int(points),
            points_per_batch=int(points_batch),
            crop_n_layers=int(crop_layers),
            min_area_px=int(min_area),
            max_objects_per_image=int(max_objects),
            tile_size=int(manual_tile),
            auto_tile_large_images=bool(auto_tile),
            auto_tile_size=int(auto_tile_size),
            auto_tile_trigger_px=int(auto_tile_trigger),
            embedding_batch_size=int(embedding_batch),
            min_cluster_size=int(min_cluster),
            cpu_fallback_on_oom=bool(cpu_fallback),
            dino_cpu_fallback_on_oom=bool(dino_cpu_fallback),
        )
        bar, status, progress = _progress_widgets()
        try:
            catalog = run_discovery(input_dir, project_dir, cfg, overwrite=overwrite, progress_cb=progress)
            st.success(f"Concluído: {len(catalog['images'])} imagens e {len(catalog['objects'])} objetos propostos.")
        except Exception as exc:
            st.exception(exc)


elif page == "2. Clusterizar bounding boxes":
    st.subheader("Clustering usando as bounding boxes já revisadas no Roboflow")
    st.write(
        "Nesta etapa o SAM não procura o objeto. Cada bounding box YOLO é recortada inteira, com contexto ao redor, "
        "e o DINOv2 + HDBSCAN agrupa esses objetos. Portanto o cetáceo não é cortado por uma borda de tile."
    )
    split_dir = st.text_input("Pasta do split YOLO", "data/roboflow/train")
    c1, c2, c3 = st.columns(3)
    with c1:
        context_scale = st.selectbox("Contexto ao redor da box", [1.0, 1.4, 1.8, 2.5, 3.0], index=2)
        min_crop = st.selectbox("Crop mínimo para objetos minúsculos (px)", [32, 48, 64, 96, 128, 192], index=2)
    with c2:
        min_cluster = st.number_input("Mínimo por cluster", 2, 100, 6)
        embedding_batch = st.selectbox("Batch DINOv2", [1, 2, 4, 8, 16, 32], index=2, key="bbox_dino_batch")
    with c3:
        dino_cpu_fallback = st.checkbox("Fallback DINOv2 para CPU", value=True, key="bbox_cpu_fallback")
        overwrite = st.checkbox("Recriar projeto", value=False, key="bbox_overwrite")

    st.info(
        "Para seu dataset, use outro Projeto local, por exemplo workspace/bbox_clusters_v1, para preservar o baseline do SAM."
    )

    if st.button("Clusterizar boxes anotadas", type="primary"):
        cfg = BoxClusteringConfig(
            context_scale=float(context_scale),
            min_crop_size_px=int(min_crop),
            min_cluster_size=int(min_cluster),
            embedding_batch_size=int(embedding_batch),
            dino_cpu_fallback_on_oom=bool(dino_cpu_fallback),
        )
        bar, status, progress = _progress_widgets()
        try:
            catalog = run_box_clustering(split_dir, project_dir, cfg, overwrite=overwrite, progress_cb=progress)
            st.success(
                f"Concluído: {len(catalog['images'])} imagens com labels e {len(catalog['objects'])} bounding boxes clusterizadas."
            )
        except Exception as exc:
            st.exception(exc)


elif page == "3. Classificar pendentes":
    st.subheader("Fila de clusters pendentes")
    catalog = _project_or_stop(project_dir)
    objects = catalog["objects"]
    pending = [o for o in objects if not o.get("ignored") and not o.get("label")]
    classified = [o for o in objects if not o.get("ignored") and o.get("label")]
    ignored = [o for o in objects if o.get("ignored")]

    m1, m2, m3 = st.columns(3)
    m1.metric("Pendentes", len(pending))
    m2.metric("Classificados", len(classified))
    m3.metric("Ignorados", len(ignored))

    if not pending:
        st.success("Não há objetos pendentes. Use 'Revisar classificados' para conferir o que já foi rotulado.")
        st.stop()

    counts = Counter(int(o.get("cluster", -1)) for o in pending)
    clusters = sorted(counts, key=lambda x: (x == -1, x))
    cluster = st.selectbox(
        "Cluster pendente",
        clusters,
        format_func=lambda x: f"Cluster {x} — {counts[x]} pendentes" + (" (outliers)" if x == -1 else ""),
    )
    group = [o for o in pending if int(o.get("cluster", -1)) == int(cluster)]

    if cluster == -1:
        st.warning("Cluster -1 contém outliers do HDBSCAN. Evite aplicar uma classe em massa sem conferir as miniaturas.")

    label = st.text_input("Nome da classe")
    b1, b2 = st.columns(2)
    if b1.button("Aplicar classe e retirar da fila", type="primary"):
        if not label.strip():
            st.warning("Informe um nome de classe.")
        else:
            n = _update_objects(project_dir, [o["id"] for o in group], label=label, ignored=False)
            st.success(f"{n} objetos classificados. O grupo saiu da fila de pendentes.")
            st.rerun()
    if b2.button("Ignorar grupo e retirar da fila"):
        n = _update_objects(project_dir, [o["id"] for o in group], ignored=True)
        st.success(f"{n} objetos movidos para ignorados.")
        st.rerun()

    st.markdown("#### Amostra")
    _show_thumbnails(project_dir, group, f"pending_{cluster}")

    st.markdown("#### Exceção individual")
    obj_id = st.selectbox("Objeto", [o["id"] for o in group], key="pending_obj")
    obj = next(o for o in group if o["id"] == obj_id)
    st.image(_render_overlay(project_dir, obj), caption=obj_id, use_container_width=True)
    individual_label = st.text_input("Classe individual", key=f"pending_label_{obj_id}")
    c1, c2 = st.columns(2)
    if c1.button("Classificar apenas este objeto"):
        if not individual_label.strip():
            st.warning("Informe uma classe individual.")
        else:
            _update_objects(project_dir, [obj_id], label=individual_label, ignored=False)
            st.rerun()
    if c2.button("Ignorar apenas este objeto"):
        _update_objects(project_dir, [obj_id], ignored=True)
        st.rerun()


elif page == "4. Revisar classificados":
    st.subheader("Revisão dos clusters já classificados")
    catalog = _project_or_stop(project_dir)
    classified = [o for o in catalog["objects"] if not o.get("ignored") and o.get("label")]
    if not classified:
        st.info("Nenhum objeto classificado ainda.")
        st.stop()

    labels = sorted({str(o["label"]) for o in classified})
    selected_label = st.selectbox("Filtrar por classe", ["Todas"] + labels)
    visible = classified if selected_label == "Todas" else [o for o in classified if o.get("label") == selected_label]
    counts = Counter(int(o.get("cluster", -1)) for o in visible)
    cluster = st.selectbox(
        "Cluster classificado",
        sorted(counts, key=lambda x: (x == -1, x)),
        format_func=lambda x: f"Cluster {x} — {counts[x]} objetos",
    )
    group = [o for o in visible if int(o.get("cluster", -1)) == int(cluster)]
    current = Counter(o.get("label") for o in group)
    st.write("Classes atuais no grupo:", dict(current))

    _show_thumbnails(project_dir, group, f"review_{selected_label}_{cluster}")

    new_label = st.text_input("Alterar classe do grupo visível", value=next(iter(current), ""))
    c1, c2, c3 = st.columns(3)
    if c1.button("Salvar nova classe do grupo", type="primary"):
        if new_label.strip():
            _update_objects(project_dir, [o["id"] for o in group], label=new_label, ignored=False)
            st.rerun()
    if c2.button("Reabrir grupo como pendente"):
        _update_objects(project_dir, [o["id"] for o in group], label=None, ignored=False)
        st.rerun()
    if c3.button("Ignorar grupo classificado"):
        _update_objects(project_dir, [o["id"] for o in group], ignored=True)
        st.rerun()

    st.markdown("#### Revisar/remover falso individual")
    obj_id = st.selectbox("Objeto classificado", [o["id"] for o in group], key="review_obj")
    obj = next(o for o in group if o["id"] == obj_id)
    st.image(_render_overlay(project_dir, obj), caption=obj_id, use_container_width=True)
    individual_label = st.text_input("Classe deste objeto", value=obj.get("label") or "", key=f"review_label_{obj_id}")
    b1, b2, b3 = st.columns(3)
    if b1.button("Salvar classe individual"):
        if individual_label.strip():
            _update_objects(project_dir, [obj_id], label=individual_label, ignored=False)
            st.rerun()
    if b2.button("Voltar objeto para pendentes"):
        _update_objects(project_dir, [obj_id], label=None, ignored=False)
        st.rerun()
    if b3.button("Marcar objeto como falso/ignorar"):
        _update_objects(project_dir, [obj_id], ignored=True)
        st.rerun()


elif page == "5. Revisar ignorados":
    st.subheader("Itens ignorados")
    catalog = _project_or_stop(project_dir)
    ignored = [o for o in catalog["objects"] if o.get("ignored")]
    if not ignored:
        st.info("Nenhum objeto ignorado.")
        st.stop()

    counts = Counter(int(o.get("cluster", -1)) for o in ignored)
    cluster = st.selectbox(
        "Cluster com itens ignorados",
        sorted(counts, key=lambda x: (x == -1, x)),
        format_func=lambda x: f"Cluster {x} — {counts[x]} ignorados",
    )
    group = [o for o in ignored if int(o.get("cluster", -1)) == int(cluster)]
    _show_thumbnails(project_dir, group, f"ignored_{cluster}")

    if st.button("Restaurar todo o grupo", type="primary"):
        _update_objects(project_dir, [o["id"] for o in group], ignored=False)
        st.rerun()

    obj_id = st.selectbox("Objeto ignorado", [o["id"] for o in group], key="ignored_obj")
    obj = next(o for o in group if o["id"] == obj_id)
    st.image(_render_overlay(project_dir, obj), caption=obj_id, use_container_width=True)
    if st.button("Restaurar apenas este objeto"):
        _update_objects(project_dir, [obj_id], ignored=False)
        st.rerun()


elif page == "6. Exportar":
    st.subheader("Exportação")
    catalog = _project_or_stop(project_dir)
    labeled = [o for o in catalog["objects"] if o.get("label") and not o.get("ignored")]
    st.metric("Objetos classificados e ativos", len(labeled))
    st.write("Classes:", sorted({o["label"] for o in labeled}))

    if catalog.get("project_type") == "bbox_clustering":
        st.info(
            "Este projeto foi criado a partir das bounding boxes existentes. Ele ainda não possui máscaras de segmentação. "
            "Por enquanto exporte o manifesto de revisão; a próxima etapa será gerar máscaras com SAM guiado pela box."
        )
        output = st.text_input("Manifesto JSON", "exports/bbox_clusters_manifest.json")
        if st.button("Exportar manifesto", type="primary"):
            result = export_bbox_manifest(project_dir, output)
            st.success(f"Manifesto exportado com {result['rows']} objetos.")
            st.json(result)
    else:
        output = st.text_input("Pasta de saída", "exports/yolo")
        if st.button("Exportar bounding boxes + segmentation", type="primary"):
            try:
                result = export_yolo(project_dir, output)
                st.success(f"Exportado: {result['images']} imagens / {result['objects']} objetos.")
                st.json(result)
            except Exception as exc:
                st.exception(exc)
        st.code("labels_detection/*.txt   # YOLO bounding box\nlabels_segmentation/*.txt # YOLO segmentation\nclasses.txt\ndata.yaml")


else:
    st.subheader("Data augmentation")
    catalog = _project_or_stop(project_dir)
    if catalog.get("project_type") == "bbox_clustering":
        st.warning(
            "O projeto de bounding boxes ainda não possui máscaras. Gere primeiro as segmentações guiadas pelas boxes antes de usar augmentation com máscaras."
        )
        st.stop()

    output = st.text_input("Pasta de saída", "exports/augmented")
    c1, c2, c3 = st.columns(3)
    copies = c1.number_input("Cópias por imagem", 1, 20, 3)
    size = c2.selectbox("Tamanho", [512, 640, 768, 1024], index=1)
    aggressive = c3.checkbox("Augmentation agressiva", value=False)
    mosaic = st.checkbox("Gerar mosaicos 2x2", value=True)
    st.write(
        "Inclui flip, rotação, shear/affine, brilho, exposição, blur, motion blur, ruído, saturação/hue, grayscale, cutout e efeitos fotométricos."
    )
    if st.button("Gerar dataset aumentado", type="primary"):
        try:
            result = augment_project(project_dir, output, int(copies), int(size), aggressive, mosaic)
            st.success(f"Geradas {result['generated_images']} imagens ({result['mosaics']} mosaicos).")
            st.json(result)
        except Exception as exc:
            st.exception(exc)
