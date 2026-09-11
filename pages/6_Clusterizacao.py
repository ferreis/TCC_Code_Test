from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextual_help import help_for, help_text

from tcc_discovery.box_clustering import BoxClusteringConfig, run_box_clustering
from tcc_discovery.dataset_augmentation import discover_yolo_datasets, resolve_yolo_split
from tcc_discovery.workspace_registry import add_entry, list_entries

st.set_page_config(page_title='TCC - Clusterização', layout='wide')
st.title('Clusterização', help=help_for('Clusterização'))
help_text('Etapa independente: dataset YOLO anotado → crops das bounding boxes → DINOv2 → HDBSCAN → revisão humana.', caption=True)

st.info(
    'Clusterização não executa augmentation nem treinamento. Depois de revisar/classificar os clusters, '
    'o dataset resultante pode seguir para divisão train/valid/test e somente então para augmentation.'
)

datasets = discover_yolo_datasets(REPO_ROOT / 'data', REPO_ROOT / 'exports')
options = [str(p) for p in datasets]
selected = st.selectbox('Dataset YOLO', options + ['Digitar caminho...'], index=0 if options else len(options), help=help_for('Dataset YOLO'))
if selected == 'Digitar caminho...':
    data_yaml = Path(st.text_input('Caminho do data.yaml', 'data/roboflow/data.yaml', help=help_for('Caminho do data.yaml')))
else:
    data_yaml = Path(selected)
if not data_yaml.is_absolute():
    data_yaml = (REPO_ROOT / data_yaml).resolve()

split_dir = data_yaml.parent / 'train'
try:
    train_images = resolve_yolo_split(data_yaml, 'train') if data_yaml.exists() else None
    if train_images is not None:
        split_dir = train_images.parent if train_images.name == 'images' else train_images
except Exception as exc:
    st.warning(f'Não foi possível resolver automaticamente o split train: {exc}')

st.text_input('Split usado para clusterização', value=str(split_dir), disabled=True, help=help_for('Split usado para clusterização'))

entries = list_entries('cluster_projects')
workspace_labels = [f"{e['name']} — {e['value']}" for e in entries]
choice = st.selectbox('Workspace de clusterização', workspace_labels + ['➕ Criar novo workspace'], help=help_for('Workspace de clusterização'))
if choice == '➕ Criar novo workspace':
    workspace_name = st.text_input('Nome do novo workspace', 'bbox_clusters_v1', help=help_for('Nome do novo workspace'))
    workspace_path = st.text_input('Pasta', f'workspace/{workspace_name}')
else:
    entry = entries[workspace_labels.index(choice)]
    workspace_name = str(entry['name'])
    workspace_path = str(entry['value'])

c1, c2, c3, c4 = st.columns(4)
context_scale = c1.selectbox('Contexto ao redor da box', [1.0, 1.4, 1.8, 2.5, 3.0], index=2, help=help_for('Contexto ao redor da box'))
min_crop = c2.selectbox('Crop mínimo (px)', [32, 48, 64, 96, 128, 192], index=2, help=help_for('Crop mínimo (px)'))
min_cluster = c3.number_input('Mínimo por cluster', 2, 100, 6, help=help_for('Mínimo por cluster'))
embedding_batch = c4.selectbox('Batch DINOv2', [1, 2, 4, 8, 16, 32], index=2, help=help_for('Batch DINOv2'))

c1, c2 = st.columns(2)
cpu_fallback = c1.checkbox('Fallback DINOv2 para CPU', value=True, help=help_for('Fallback DINOv2 para CPU'))
overwrite = c2.checkbox('Recriar workspace se já existir', value=False, help=help_for('Recriar workspace se já existir'))

if st.button('Executar clusterização', type='primary', help=help_for('Executar clusterização')):
    try:
        if not data_yaml.exists():
            raise FileNotFoundError(data_yaml)
        if not split_dir.exists():
            raise FileNotFoundError(f'Split train não encontrado: {split_dir}')
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
