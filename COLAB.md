# Google Colab

O projeto usa um único notebook:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
```

O notebook não possui mais uma cópia própria dos hiperparâmetros. A fonte única é:

```text
configs/runtime.yaml
```

O Colab usa o perfil:

```text
colab
```

e o computador local/Streamlit usa normalmente:

```text
local
```

Ambos executam o mesmo `scripts/train_experiments.py`.

## Preparação

1. Abra o notebook no Colab.
2. Selecione GPU.
3. Configure em Secrets `GITHUB_TOKEN` e `ROBOFLOW_API_KEY`; `NGROK_AUTHTOKEN` só é necessário se quiser abrir Streamlit por túnel.
4. Execute as células em ordem.

O notebook preserva o PyTorch/CUDA do Colab, instala `requirements-colab.txt`, instala Tencent/YOLO-Master e valida `ultralytics.utils.lora` antes do treino.

## Configuração compartilhada

Edite `configs/runtime.yaml` para alterar:

```text
modelos YOLO26
caminhos dos datasets
epochs
imgsz
batch/workers/cache por perfil
AMP
optimizer
patience
seed
staging local de dataset
rank/alpha/dropout da LoRA
Roboflow geral e especialistas
clustering
SAM
caminhos de resultados/exportação
```

`configs/training_experiments.yaml` define apenas a topologia dos experimentos e as relações entre Modelo Geral, LoRAs e controles Full Fine-Tune.

## Persistência

O perfil `colab` define:

```text
persist_root: /content/drive/MyDrive/TCC_Code_Test_runtime
```

O notebook liga:

```text
data/
workspace/
exports/
weights/
```

a esse diretório do Drive.

## Staging para acelerar treinamento

No perfil `colab`:

```yaml
use_local_dataset_cache: true
local_data_root: /content/TCC_training_data
```

Antes de uma execução real, a CLI copia somente o dataset da etapa selecionada para o SSD local do runtime. O treino lê de `/content`; checkpoints e resultados continuam persistindo no Drive.

## Datasets Roboflow

O dataset geral e os especialistas ficam configurados em `configs/runtime.yaml`.

Exemplo de especialista:

```yaml
det_lora_01:
  project: slug-do-projeto-no-roboflow
  version: 1
  output: data/specialists/detection/lora_01
  download: true
  overwrite: false
```

Quando `download: true`, o notebook baixa o dataset antes do treinamento.

## Treinamento

O helper do notebook apenas chama a mesma CLI do ambiente local:

```python
run_stage('smoke_det_yolo26n')
run_stage('det_general')
run_stage('det_lora_01')
run_stage('det_fullft_01')
```

Equivalente no terminal:

```bash
python scripts/train_experiments.py --profile colab --only det_general
```

Cada LoRA parte novamente do `best.pt` do Modelo Geral. Os adapters não são treinados em sequência e não são fundidos durante o protocolo principal.

## Exportação para TCC_Analise

Após cada treino real, a CLI atualiza:

```text
MyDrive/TCC_Code_Test_runtime/exports/
├── TCC_Analise_models/
└── TCC_Analise_models.zip
```

O ZIP é organizado para ser extraído diretamente em `TCC_Analise/models/`.

## Resultados

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

O relatório registra parâmetros, checkpoint geral de origem, tamanho da LoRA, GPU, VRAM, tempo, métricas e velocidade.

Mais detalhes: `TRAINING.md`.
