from __future__ import annotations

from collections import Counter
from pathlib import Path

import streamlit as st

from tcc_discovery.augment import augment_project
from tcc_discovery.core import (
    DiscoveryConfig,
    export_yolo,
    load_project,
    render_object_overlay,
    run_discovery,
    set_cluster_label,
    set_object_label,
)

st.set_page_config(page_title="TCC - Object Discovery Lab", layout="wide")
st.title("Object Discovery Lab")
st.caption("SAM gera máscaras/boxes sem classe; DINOv2 + HDBSCAN agrupa objetos semelhantes; você rotula depois.")

page = st.sidebar.radio("Etapa", ["1. Descobrir objetos", "2. Classificar clusters", "3. Exportar YOLO", "4. Aumentar dataset"])
project_dir = Path(st.sidebar.text_input("Projeto local", "workspace/discovery"))

if page == "1. Descobrir objetos":
    st.subheader("Descoberta não supervisionada")
    input_dir = st.text_input("Pasta de imagens", "data/roboflow")
    checkpoint = st.text_input("Checkpoint SAM", "weights/sam_vit_b_01ec64.pth")
    c1, c2, c3 = st.columns(3)
    with c1:
        points = st.select_slider("SAM points_per_side", [16, 24, 32, 48, 64], value=32)
        min_area = st.number_input("Área mínima da máscara (px)", 10, 100000, 120)
    with c2:
        max_objects = st.number_input("Máx. objetos/imagem", 1, 300, 40)
        tile_size = st.selectbox("Tile para objetos pequenos", [0, 768, 1024, 1280, 1536, 2048], index=0)
    with c3:
        min_cluster = st.number_input("Mínimo por cluster", 2, 100, 6)
        overwrite = st.checkbox("Recriar projeto", value=False)
    st.info("Para imagens aéreas grandes e animais pequenos, teste tile 1024 ou 1536. Isso é mais lento, mas costuma recuperar objetos pequenos.")
    if st.button("Executar descoberta", type="primary"):
        cfg = DiscoveryConfig(
            sam_checkpoint=checkpoint,
            points_per_side=int(points),
            min_area_px=int(min_area),
            max_objects_per_image=int(max_objects),
            tile_size=int(tile_size),
            min_cluster_size=int(min_cluster),
        )
        bar = st.progress(0.0)
        status = st.empty()

        def progress(done, total, message):
            if total:
                bar.progress(min(1.0, done / total))
            status.write(message)

        try:
            catalog = run_discovery(input_dir, project_dir, cfg, overwrite=overwrite, progress_cb=progress)
            st.success(f"Concluído: {len(catalog['images'])} imagens e {len(catalog['objects'])} objetos propostos.")
        except Exception as exc:
            st.exception(exc)

elif page == "2. Classificar clusters":
    st.subheader("Classificação humana dos grupos")
    if not (project_dir / "catalog.json").exists():
        st.warning("Execute a descoberta primeiro.")
        st.stop()
    catalog = load_project(project_dir)
    objects = catalog["objects"]
    if not objects:
        st.warning("Nenhum objeto encontrado.")
        st.stop()
    counts = Counter(int(o.get("cluster", -1)) for o in objects)
    clusters = sorted(counts, key=lambda x: (x == -1, x))
    cluster = st.selectbox("Cluster", clusters, format_func=lambda x: f"Cluster {x} — {counts[x]} objetos" + (" (outliers)" if x == -1 else ""))
    group = [o for o in objects if int(o.get("cluster", -1)) == int(cluster)]
    existing_labels = Counter(o.get("label") for o in group if o.get("label"))
    if existing_labels:
        st.write("Rótulos atuais:", dict(existing_labels))

    label = st.text_input("Nome da classe", value=next(iter(existing_labels), ""))
    b1, b2 = st.columns(2)
    if b1.button("Aplicar classe ao cluster", type="primary"):
        n = set_cluster_label(project_dir, cluster, label, ignored=False)
        st.success(f"Classe aplicada a {n} objetos.")
        st.rerun()
    if b2.button("Ignorar cluster"):
        n = set_cluster_label(project_dir, cluster, None, ignored=True)
        st.success(f"{n} objetos marcados como ignorados.")
        st.rerun()

    st.markdown("#### Amostra do cluster")
    show_n = st.slider("Quantidade de miniaturas", 4, min(60, len(group)), min(20, len(group)))
    cols = st.columns(5)
    for i, obj in enumerate(group[:show_n]):
        crop = project_dir / obj["crop_path"]
        with cols[i % 5]:
            st.image(str(crop), caption=f"{obj['id']}\nlabel={obj.get('label')}", use_container_width=True)

    st.markdown("#### Revisão individual / exceção")
    obj_id = st.selectbox("Objeto", [o["id"] for o in group])
    obj = next(o for o in group if o["id"] == obj_id)
    st.image(render_object_overlay(project_dir, obj), caption=obj_id, use_container_width=True)
    individual_label = st.text_input("Classe individual", value=obj.get("label") or "", key=f"individual_{obj_id}")
    c1, c2 = st.columns(2)
    if c1.button("Salvar classe individual"):
        set_object_label(project_dir, obj_id, individual_label, ignored=False)
        st.success("Objeto atualizado.")
        st.rerun()
    if c2.button("Ignorar objeto"):
        set_object_label(project_dir, obj_id, None, ignored=True)
        st.success("Objeto ignorado.")
        st.rerun()

elif page == "3. Exportar YOLO":
    st.subheader("Exportação")
    if not (project_dir / "catalog.json").exists():
        st.warning("Projeto não encontrado.")
        st.stop()
    catalog = load_project(project_dir)
    labeled = [o for o in catalog["objects"] if o.get("label") and not o.get("ignored")]
    st.metric("Objetos rotulados", len(labeled))
    st.write("Classes:", sorted({o["label"] for o in labeled}))
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
    st.subheader("Data augmentation com máscaras preservadas")
    if not (project_dir / "catalog.json").exists():
        st.warning("Projeto não encontrado.")
        st.stop()
    output = st.text_input("Pasta de saída", "exports/augmented")
    c1, c2, c3 = st.columns(3)
    copies = c1.number_input("Cópias por imagem", 1, 20, 3)
    size = c2.selectbox("Tamanho", [512, 640, 768, 1024], index=1)
    aggressive = c3.checkbox("Augmentation agressiva", value=False)
    mosaic = st.checkbox("Gerar mosaicos 2x2", value=True)
    st.write(
        "Inclui flip, 90° rotate, rotação, shear/affine, brilho, exposição/gamma, blur, motion blur, ruído, saturação/hue, grayscale, cutout e efeitos fotométricos dentro do objeto."
    )
    if st.button("Gerar dataset aumentado", type="primary"):
        try:
            result = augment_project(project_dir, output, int(copies), int(size), aggressive, mosaic)
            st.success(f"Geradas {result['generated_images']} imagens ({result['mosaics']} mosaicos).")
            st.json(result)
        except Exception as exc:
            st.exception(exc)
