from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcc_discovery.dataset_augmentation import (
    DEFAULT_AUGMENTATIONS,
    augment_yolo_detection_dataset,
    discover_yolo_datasets,
    inspect_yolo_detection_dataset,
)
from tcc_discovery.runtime_config import load_runtime_config, save_runtime_config
from tcc_discovery.workspace_registry import add_entry, list_entries

st.set_page_config(page_title='TCC - Augmentation', layout='wide')
st.title('Data Augmentation')
st.caption('Etapa independente para datasets YOLO já divididos. Somente o TRAIN é aumentado; VALID e TEST são preservados sem alteração.')

st.warning(
    'Regra experimental: augmentation nunca é aplicada em validation/test. Esta página copia esses splits sem modificá-los '
    'e gera novas imagens apenas em train.'
)

found = discover_yolo_datasets(REPO_ROOT / 'data', REPO_ROOT / 'exports', REPO_ROOT / 'workspace')
registered = list_entries('datasets')
source_options: list[tuple[str, str]] = []
seen: set[str] = set()
for entry in registered:
    value = str(entry.get('value', ''))
    if value and value not in seen:
        source_options.append((f"{entry.get('name', 'dataset')} — {value}", value))
        seen.add(value)
for path in found:
    value = str(path)
    if value not in seen:
        source_options.append((value, value))
        seen.add(value)
labels = [x[0] for x in source_options]
choice = st.selectbox('Dataset de entrada', labels + ['Digitar/importar caminho...'], index=0 if labels else len(labels))
if choice == 'Digitar/importar caminho...':
    source_yaml = st.text_input('Caminho do data.yaml', 'data/roboflow/data.yaml')
else:
    source_yaml = source_options[labels.index(choice)][1]

source_preview = Path(source_yaml)
if not source_preview.is_absolute():
    source_preview = (REPO_ROOT / source_preview).resolve()

source_ok = False
try:
    dataset_info = inspect_yolo_detection_dataset(source_preview)
    source_ok = bool(dataset_info['exists']['train'])
    st.markdown('#### Dataset detectado')
    m1, m2, m3 = st.columns(3)
    m1.metric('Train', dataset_info['images']['train'])
    m2.metric('Validation', dataset_info['images']['val'])
    m3.metric('Test', dataset_info['images']['test'])
    with st.expander('Ver caminhos resolvidos'):
        st.write('data.yaml:', dataset_info['data_yaml'])
        st.write('TRAIN:', dataset_info['resolved']['train'])
        st.write('VALID:', dataset_info['resolved']['val'])
        st.write('TEST:', dataset_info['resolved']['test'])
        st.caption(
            'Exports Roboflow com caminhos como ../train/images são corrigidos automaticamente quando '
            'as pastas train/valid/test estão ao lado do data.yaml.'
        )
    if not source_ok:
        st.error('O split TRAIN ainda não pôde ser localizado. Confira o data.yaml e a estrutura de pastas.')
except Exception as exc:
    dataset_info = None
    st.error(f'Não foi possível validar o dataset selecionado: {exc}')

out_entries = list_entries('augmentation_outputs')
out_labels = [f"{e['name']} — {e['value']}" for e in out_entries]
out_choice = st.selectbox('Dataset de saída', out_labels + ['➕ Criar novo dataset aumentado'])
if out_choice == '➕ Criar novo dataset aumentado':
    dataset_name = st.text_input('Nome do dataset aumentado', 'roboflow_augmented_v1')
    output_dir = st.text_input('Pasta de saída', f'data/augmented/{dataset_name}')
else:
    entry = out_entries[out_labels.index(out_choice)]
    dataset_name = str(entry['name'])
    output_dir = str(entry['value'])

c1, c2, c3, c4 = st.columns(4)
copies = c1.number_input('Cópias aumentadas por imagem de treino', 1, 20, 3)
size = c2.selectbox('Tamanho final', [512, 640, 768, 1024, 1280], index=1)
aggressive = c3.checkbox('Augmentation agressiva', value=True)
mosaic = c4.checkbox('Gerar mosaicos 2x2', value=True)

seed = st.number_input('Seed', 0, 2_147_483_647, 42)
overwrite = st.checkbox('Recriar dataset de saída', value=False)

st.markdown('#### Operações')
cols = st.columns(4)
enabled: dict[str, bool] = {}
labels_pt = {
    'horizontal_flip': 'Flip horizontal',
    'vertical_flip': 'Flip vertical',
    'rotate90': 'Rotação 90°',
    'rotate': 'Rotação livre',
    'affine': 'Affine / shear / escala',
    'brightness_contrast': 'Brilho e contraste',
    'gamma': 'Gamma / exposição',
    'hsv': 'Hue / saturação / valor',
    'gaussian_blur': 'Gaussian blur',
    'motion_blur': 'Motion blur',
    'gaussian_noise': 'Ruído gaussiano',
    'grayscale': 'Grayscale',
    'coarse_dropout': 'Cutout / coarse dropout',
}
for i, key in enumerate(DEFAULT_AUGMENTATIONS):
    with cols[i % 4]:
        enabled[key] = st.checkbox(labels_pt.get(key, key), value=True, key=f'aug_{key}')

st.markdown('#### Proteções de split')
a, b, c = st.columns(3)
a.checkbox('Manter imagens originais de train', value=True, disabled=True)
b.checkbox('Preservar validation sem augmentation', value=True, disabled=True)
c.checkbox('Preservar test sem augmentation', value=True, disabled=True)

if st.button('Gerar dataset aumentado', type='primary', disabled=not source_ok):
    try:
        source = Path(source_yaml)
        if not source.is_absolute():
            source = (REPO_ROOT / source).resolve()
        output = Path(output_dir)
        if not output.is_absolute():
            output = (REPO_ROOT / output).resolve()

        st.markdown('#### Progresso')
        progress_bar = st.progress(0.0, text='Preparando augmentation...')
        time_box = st.empty()
        terminal_box = st.empty()
        terminal_lines: deque[str] = deque(maxlen=18)
        started = time.monotonic()

        def fmt_seconds(value: float) -> str:
            value = max(0, int(round(value)))
            hours, rem = divmod(value, 3600)
            minutes, seconds = divmod(rem, 60)
            if hours:
                return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
            return f'{minutes:02d}:{seconds:02d}'

        def log_cb(message: str) -> None:
            elapsed = time.monotonic() - started
            terminal_lines.append(f'[{fmt_seconds(elapsed)}] {message}')
            terminal_box.code('\n'.join(terminal_lines), language='text')

        def progress_cb(done: int, total: int, message: str) -> None:
            ratio = min(1.0, max(0.0, done / total if total else 0.0))
            elapsed = time.monotonic() - started
            eta = (elapsed / ratio - elapsed) if ratio > 0 else 0.0
            progress_bar.progress(ratio, text=f'{ratio * 100:5.1f}% · {message}')
            eta_text = fmt_seconds(eta) if ratio > 0 else 'calculando...'
            time_box.caption(
                f'Decorrido: {fmt_seconds(elapsed)} · ETA aproximada: {eta_text} · '
                f'Operações: {done}/{total}'
            )

        log_cb('Iniciando processamento. A tela será atualizada automaticamente; não é necessário clicar em atualizar.')
        result = augment_yolo_detection_dataset(
            source,
            output,
            copies_per_image=int(copies),
            size=int(size),
            aggressive=bool(aggressive),
            include_mosaic=bool(mosaic),
            include_original_train=True,
            preserve_validation=True,
            preserve_test=True,
            enabled_augmentations=enabled,
            seed=int(seed),
            overwrite=bool(overwrite),
            progress_cb=progress_cb,
            log_cb=log_cb,
        )
        progress_bar.progress(1.0, text='100% · Augmentation concluída')
        time_box.caption(f'Tempo total: {fmt_seconds(float(result.get("duration_seconds", 0.0)))}')

        output_yaml = str(output / 'data.yaml')
        add_entry('augmentation_outputs', dataset_name, str(output))
        add_entry('datasets', dataset_name, output_yaml, metadata={'source': str(source)})
        st.success('Dataset aumentado criado.')
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Train original', result['source_train_images'])
        m2.metric('Novas augmentations', result['generated_augmented_train'])
        m3.metric('Mosaicos', result['generated_mosaics_train'])
        m4.metric('Train final', result['final_train_images'])
        st.write('Validation preservado:', result['validation_images_unchanged'])
        st.write('Test preservado:', result['test_images_unchanged'])

        b1, b2, b3 = st.columns(3)
        b1.metric('Boxes lidas no train', result.get('bbox_labels_total', 0))
        b2.metric('Boxes corrigidas na borda', result.get('bbox_labels_corrected_at_border', 0))
        b3.metric('Boxes inválidas ignoradas', result.get('bbox_labels_skipped_invalid', 0))
        if result.get('bbox_labels_corrected_at_border', 0):
            st.info(
                'Algumas boxes encostavam na borda da imagem e tinham pequenas diferenças de arredondamento. '
                'Elas foram limitadas a [0,1] antes da augmentation.'
            )
        if result.get('bbox_labels_skipped_invalid', 0):
            st.warning(
                'Foram encontradas boxes claramente inválidas e elas não foram usadas nas imagens aumentadas. '
                'A quantidade fica registrada em augmentation_manifest.json.'
            )

        st.code(output_yaml)
        st.session_state['last_augmented_yaml'] = output_yaml
    except Exception as exc:
        st.exception(exc)

if st.session_state.get('last_augmented_yaml'):
    st.divider()
    st.markdown('#### Enviar para o treinamento')
    st.write('O dataset abaixo já está pronto para ser selecionado no treinamento:')
    st.code(st.session_state['last_augmented_yaml'])
    if st.button('Usar este dataset como det_general em configs/runtime.yaml'):
        runtime_path = REPO_ROOT / 'configs' / 'runtime.yaml'
        runtime = load_runtime_config(runtime_path)
        value = Path(st.session_state['last_augmented_yaml'])
        try:
            value = value.relative_to(REPO_ROOT)
        except ValueError:
            pass
        runtime.setdefault('datasets', {})['det_general'] = str(value)
        save_runtime_config(runtime, runtime_path)
        st.success('det_general atualizado. A página Treinamento usará este data.yaml.')
