# TCC Code Test — Object Discovery & Training Lab

Protótipo para descoberta, agrupamento, revisão, segmentação e treinamento de modelos em imagens aéreas de cetáceos.

O projeto possui três frentes:

1. **Baseline automático:** imagem → SAM → propostas/máscaras → DINOv2 → HDBSCAN.
2. **Fluxo para dataset anotado:** bounding boxes Roboflow → crops → DINOv2 → HDBSCAN → revisão humana → SAM guiado pela box → segmentação.
3. **Treinamento modular YOLO26:** um Modelo Geral em dataset amplo + LoRAs especialistas independentes em datasets pequenos + controles Full Fine-Tune equivalentes.

## Instalação local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Para experimentos LoRA, o Streamlit oferece a instalação do Tencent/YOLO-Master; também é possível executar:

```bash
python scripts/setup_yolo_master.py --dir .yolo-master
```

## Configuração única

Todos os parâmetros editáveis de treinamento ficam em:

```text
configs/runtime.yaml
```

Esse arquivo é usado pelo terminal, pelo frontend Streamlit e pelo notebook Colab. Existem dois perfis iniciais:

```text
local  → GPU ~6 GB, batch inicial 8
colab  → Tesla T4 ~15 GB, batch inicial 32 e dataset em SSD local /content
```

O arquivo `configs/training_experiments.yaml` define somente a topologia das etapas; não duplica hiperparâmetros.

## Interface Streamlit

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True streamlit run app.py
```

Além das páginas de descoberta/revisão, o diretório `pages/` adiciona a página **Treinamento**.

Na página de treinamento é possível:

```text
selecionar perfil local/colab
selecionar Smoke / Modelo Geral / LoRA / Full Fine-Tune
editar e salvar configs/runtime.yaml
visualizar GPU, VRAM e GPU-Util
iniciar treinamento em segundo plano
acompanhar o log
parar a execução
consultar resultados já registrados
preparar YOLO-Master quando uma LoRA for selecionada
```

O frontend não implementa outro treinador: ele chama exatamente `scripts/train_experiments.py`.

## Dataset Roboflow

Crie `.env` a partir de `.env.example` e configure sua chave. Download manual do dataset geral:

```bash
python scripts/download_roboflow.py --format yolo26 --output data/roboflow --overwrite
```

Os projetos Roboflow do Modelo Geral e das LoRAs também podem ser declarados em `configs/runtime.yaml`.

## Treinamento pelo terminal

Smoke local:

```bash
python scripts/train_experiments.py --profile local --only smoke_det_yolo26n
```

Modelo Geral:

```bash
python scripts/train_experiments.py --profile local --only det_general
```

LoRA 01:

```bash
python scripts/train_experiments.py --profile local --only det_lora_01
```

Controle Full Fine-Tune 01:

```bash
python scripts/train_experiments.py --profile local --only det_fullft_01
```

Listar etapas:

```bash
python scripts/train_experiments.py --profile local --group all --list
```

## Arquitetura LoRA

```text
Dataset amplo
    ↓
Modelo Geral
    ↓
best.pt
    ├── LoRA 01 ← dataset pequeno 01
    ├── LoRA 02 ← dataset pequeno 02
    ├── LoRA 03 ← dataset pequeno 03
    ├── LoRA 04 ← dataset pequeno 04
    └── LoRA 05 ← dataset pequeno 05
```

Cada LoRA começa novamente no mesmo checkpoint geral. `LoRA 02` não continua de `LoRA 01`.

Para comparar custo × desempenho, cada especialista N pode ter:

```text
LoRA N           → Modelo Geral + adapter treinável
Full Fine-Tune N → mesmo Modelo Geral + mesmos dados, todos os pesos liberados
```

## Google Colab

Notebook único:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
```

O notebook não contém mais uma célula `PARAMS` duplicada. Depois do bootstrap, ele lê `configs/runtime.yaml`, usa o perfil `colab` e chama a mesma CLI do computador local.

No Colab, configure Secrets:

```text
GITHUB_TOKEN
ROBOFLOW_API_KEY
NGROK_AUTHTOKEN   # opcional
```

Detalhes em `COLAB.md`.

## Staging de dataset

No perfil `colab`, o dataset persistido no Drive é copiado para SSD local antes da etapa selecionada:

```text
Google Drive → /content/TCC_training_data/<etapa> → GPU
```

No perfil local isso começa desativado porque os dados já estão em disco local.

## Resultados e exportação

Cada etapa atualiza:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

Após treinamento real, também é criado/atualizado:

```text
exports/TCC_Analise_models/
exports/TCC_Analise_models.zip
```

O ZIP é estruturado para ser extraído diretamente dentro de `TCC_Analise/models/`.

## Clustering

O fluxo recomendado para as bounding boxes existentes é:

```text
Bounding boxes Roboflow
        ↓
crops completos + contexto
        ↓
DINOv2
        ↓
HDBSCAN
        ↓
revisão humana
        ↓
dataset especialista opcional
        ↓
LoRA
```

O cluster não é uma LoRA por si só; ele pode ser usado para selecionar e organizar exemplos que formarão um dataset especialista.

## Inferência modular a avaliar

```text
A) Modelo Geral + uma LoRA selecionada
B) Modelo Geral → ROI candidato → Modelo Geral + LoRA especialista
C) Modelo Geral → vários especialistas independentes → fusão espacial
```

A estratégia B é a mais próxima da hipótese de localizar primeiro com conhecimento geral e depois reavaliar com um especialista.

## Regras experimentais

- Não misturar `train`, `valid` e `test`.
- Para vídeo, dividir preferencialmente por vídeo ou blocos temporais.
- Aplicar augmentation somente em treino.
- Não usar `test` para selecionar hiperparâmetros.
- Comparar cada LoRA com Full Fine-Tune usando o mesmo checkpoint geral e o mesmo dataset pequeno.
- Não usar cluster semântico para fundir detecções de tiles; a fusão deve ser espacial no sistema global da imagem.
