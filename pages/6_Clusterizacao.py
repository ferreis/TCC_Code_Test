from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcc_discovery.box_clustering import BoxClusteringConfig, run_box_clustering
from tcc_discovery.dataset_augmentation import discover_yolo_datasets
from tcc_discovery.workspace_registry import add_entry, list_entries

st.set_page_config(page_title='TCC - Clusterização', layout='wide')
st.title('Clusterização')
st.caption('Etapa independente: dataset YOLO anotado → crops das bounding boxes → DINOv2 → HDBSCAN → revisão humana.')

st.info(
    'Clusterização não executa augmentation nem treinamento. Depois de revisar/classificar os clusters, '
    'o dataset resultante pode seguir para divisão train/valid/test e somente então para augmentation.'
)

datasets = discover_yolo_datasets(REPO_ROOT / 'data', REPO_ROOT / 'exports')
options = [str(p) for p in datasets]
selected = st.selectbox('Dataset YOLO', options + ['Digitar caminho...'], index=0 if options else len(options))
if selected == 'Digitar caminho...':
    data_yaml = Path(st.text_input('Caminho do data.yaml', 'data/roboflow/data.yaml'))
else:
    data_yaml = Path(selected)

split_dir = data_yaml.parent / 'train'
if data_yaml.exists():
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    raw_train = cfg.get('train')
    if isinstance(raw_train, str):
        root = Path(str(cfg.get('path', '.')))
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        train_images = Path(raw_train)
        if not train_images.is_absolute():
            train_images = (root / train_images).resolve()
        split_dir = train_images.parent if train_images.name == 'images' else train_images

st.text_input('Split usado para clusterização', value=str(split_dir), disabled=True)

entries = list_entries('cluster_projects')
workspace_labels = [f"{e['name']} — {e['value']}" for e in entries]
choice = st.selectbox('Workspace de clusterização', workspace_labels + ['➕ Criar novo workspace'])
if choice == '➕ Criar novo workspace':
    workspace_name = st.text_input('Nome do novo workspace', 'bbox_clusters_v1')
    workspace_path = st.text_input('Pasta', f'workspace/{workspace_name}')
else:
    entry = entries[workspace_labels.index(choice)]
    workspace_name = str(entry['name'])
    workspace_path = str(entry['value'])

c1, c2, c3, c4 = st.columns(4)
context_scale = c1.selectbox('Contexto ao redor da box', [1.0, 1.4, 1.8, 2.5, 3.0], index=2)
min_crop = c2.selectbox('Crop mínimo (px)', [32, 48, 64, 96, 128, 192], index=2)
min_cluster = c3.number_input('Mínimo por cluster', 2, 100, 6)
embedding_batch = c4.selectbox('Batch DINOv2', [1, 2, 4, 8, 16, 32], index=2)

c1, c2 = st.columns(2)
cpu_fallback = c1.checkbox('Fallback DINOv2 para CPU', value=True)
overwrite = c2.checkbox('Recriar workspace se já existir', value=False)

if st.button('Executar clusterização', type='primary'):
    try:
        if not data_yaml.exists():
            raise FileNotFoundError(data_yaml)
        add_entry('cluster_projects', workspace_name, workspace_path, metadata={'data_yaml': str(data_yaml)})
        cfg = BoxClusteringConfig(
            context_scale=float(context_scale),
            min_crop_size_px=int(min_crop),
            min_cluster_size=int(min_cluster),
            embedding_batch_size=int(embedding_batch),
            dino_cpu_fallback_on_oom=bool(cpu_fallback),
        )
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress(done, total, message):
            if total:
                progress_bar.progress(min(1.0, done / total))
            status.write(message)

        catalog = run_box_clustering(split_dir, workspace_path, cfg, overwrite=overwrite, progress_cb=progress)
        progress_bar.progress(1.0)
        st.success(f"Clusterização concluída: {len(catalog.get('objects', []))} objetos.")
        st.write('Workspace salvo em:', workspace_path)
    except Exception as exc:
        st.exception(exc)

st.divider()
st.markdown('**Próxima etapa:** revisão/classificação dos clusters. A augmentation fica separada e só deve receber um dataset já dividido em train/valid/test.')
try:
    st.page_link('app.py', label='Abrir revisão/classificação no Object Discovery Lab')
except Exception:
    pass
