# TCC Code Test — Object Discovery & Training Lab

Protótipo para descoberta, agrupamento, revisão, segmentação e treinamento de modelos em imagens aéreas de cetáceos.

O projeto possui três frentes complementares:

1. **Baseline automático:** imagem → SAM → propostas/máscaras → DINOv2 → HDBSCAN.
2. **Fluxo recomendado para o dataset anotado:** bounding boxes Roboflow → crops completos → DINOv2 → HDBSCAN → revisão humana → SAM guiado pela box → segmentação.
3. **Treinamento modular YOLO26:** um Modelo Geral treinado em dataset amplo e vários adapters LoRA especialistas treinados separadamente em datasets pequenos, com controles Full Fine-Tune equivalentes.

## Instalação local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Roboflow

Crie `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Preencha:

```env
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=rafael-fernando-dos-reis-mecabo
ROBOFLOW_PROJECT=detecao-de-cetaceos-afrzs-iqey9
ROBOFLOW_VERSION=1
```

Baixe no formato YOLO26:

```bash
python scripts/download_roboflow.py --format yolo26 --output data/roboflow --overwrite
```

Estrutura esperada:

```text
data/roboflow/
├── train/images + labels
├── valid/images + labels
├── test/images + labels
└── data.yaml
```

## SAM

```bash
python scripts/download_sam.py
```

Checkpoint padrão:

```text
weights/sam_vit_b_01ec64.pth
```

## Interface de descoberta/revisão

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True streamlit run app.py
```

A interface permite upload de imagem/vídeo, baseline SAM, clustering de bounding boxes, classificação de clusters, revisão de classificados/ignorados, exportação e augmentation.

## Clustering das bounding boxes

O fluxo recomendado para as anotações existentes é:

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
```

Configuração inicial:

```text
Contexto: 1.8
Crop mínimo: 64 px
Mínimo por cluster: 6
Fallback CPU: ligado
```

## Treinamento YOLO26 — nomenclatura

A nomenclatura antiga `Base | Base + LoRA | Full` foi removida. Ela sugeria três treinamentos paralelos sobre o mesmo dataset e não representava a hipótese do TCC.

Agora usamos:

```text
Modelo Geral
LoRA Especialista 01..05
Full Fine-Tune Controle 01..05
Smoke Test YOLO26n
```

### 1. Modelo Geral

Primeiro treinamos o detector em um dataset amplo:

```text
data/roboflow/data.yaml
        ↓
det_general
        ↓
workspace/training_runs/det_general_yolo26n/weights/best.pt
```

Esse `best.pt` é o checkpoint raiz para todos os especialistas de detecção.

Para segmentação, depois que o dataset estiver pronto:

```text
seg_general
```

### 2. LoRAs especialistas

Cada LoRA é um adapter independente. Todas partem do mesmo checkpoint geral, mas cada uma pode usar um dataset pequeno diferente:

```text
det_general/best.pt
    ├── det_lora_01 ← dataset pequeno 01
    ├── det_lora_02 ← dataset pequeno 02
    ├── det_lora_03 ← dataset pequeno 03
    ├── det_lora_04 ← dataset pequeno 04
    └── det_lora_05 ← dataset pequeno 05
```

`det_lora_02` não continua de `det_lora_01`. Cada treinamento começa novamente no `best.pt` do Modelo Geral.

O artefato principal é o adapter:

```text
workspace/training_runs/det_lora_01/lora_adapter/
```

O treinamento não chama `merge_lora()`; preservamos o modelo geral e os adapters separados.

### 3. Full Fine-Tune de controle

Para medir se LoRA realmente economiza recursos sem perder desempenho, cada LoRA N pode ser comparada com um Full Fine-Tune N:

```text
mesmo checkpoint geral
+ mesmo dataset pequeno N
+ mesmos hiperparâmetros comuns
```

A diferença é apenas:

```text
LoRA N           → treina adapter de baixa dimensão
Full Fine-Tune N → libera todos os pesos
```

Exemplos de IDs:

```text
det_fullft_01 ... det_fullft_05
seg_fullft_01 ... seg_fullft_05
```

## YOLO26n

YOLO26n continua no projeto como escala inicial e smoke test para a Tesla T4. Depois de estabilizar o protocolo, o Modelo Geral pode ser repetido com uma escala maior se houver justificativa e VRAM suficiente.

## Google Colab

Existe um único notebook:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
```

O notebook executa as etapas separadamente. Ele possui um helper `run_stage()` para rodar **uma etapa por vez**:

```python
run_stage('smoke_det_yolo26n')
run_stage('det_general')
run_stage('det_lora_01')
run_stage('det_fullft_01')
```

Os templates LoRA/Full FT ficam desativados no YAML versionado para evitar execução acidental. O notebook cria uma cópia de runtime e ativa apenas o ID escolhido.

O Colab usa:

```text
MyDrive/TCC_Code_Test_runtime/
├── data/
├── workspace/
├── exports/
└── weights/
```

Secrets:

```text
GITHUB_TOKEN
ROBOFLOW_API_KEY
NGROK_AUTHTOKEN   # opcional
```

Detalhes: `COLAB.md` e `TRAINING.md`.

## Como a LoRA entra na inferência

LoRA não é um modelo completo que recebe diretamente as boxes produzidas pelo Modelo Geral. O adapter modifica camadas internas do mesmo modelo-base. Para representar a ideia de um especialista posterior ao detector geral, vamos comparar:

```text
A) Modelo Geral + uma LoRA selecionada

B) Modelo Geral
       ↓
   ROI candidato
       ↓
   Modelo Geral + LoRA especialista

C) Modelo Geral
       ↓
   vários especialistas independentes
       ↓
   fusão espacial das detecções
```

A estratégia B é a implementação mais fiel da ideia de "primeiro o modelo geral, depois o vínculo especialista". O segundo estágio é uma nova inferência sobre o ROI usando o mesmo checkpoint geral com o adapter escolhido.

O YOLO-Master também possui MoLoRA, mas isso é outro experimento: os especialistas são treinados dentro de uma arquitetura roteada. Não é equivalente a treinar cinco LoRAs independentes e simplesmente juntá-las depois.

## Resultados do treinamento

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

O relatório registra hiperparâmetros, checkpoint geral de origem, parâmetros totais e treináveis, parâmetros LoRA, tamanho do adapter, GPU, VRAM, tempo, métricas e velocidade.

## Arquitetura alvo do TCC

```text
Bounding boxes revisadas no Roboflow
              |
              +--> DINOv2 --> HDBSCAN --> revisão humana
              |
              +--> Modelo Geral YOLO26
              |        |
              |        +--> auditoria de possíveis objetos não anotados
              |        +--> inferência em tiles + fusão espacial
              |        +--> LoRAs especialistas independentes
              |
              +--> SAM guiado pela box --> máscaras --> dataset de segmentação
                       |
                       +--> Modelo Geral YOLO26-seg
                                 |
                                 +--> LoRAs de segmentação independentes
```

## Regras experimentais

- Não misturar `train`, `valid` e `test`.
- Para vídeo, dividir preferencialmente por vídeo ou blocos temporais.
- Manter o baseline SAM automático como comparação.
- Aplicar augmentation somente no treino.
- Não usar `test` para selecionar hiperparâmetros.
- Comparar cada LoRA com um Full Fine-Tune usando o mesmo checkpoint geral e o mesmo dataset pequeno.
- Não usar cluster semântico para fundir detecções de tiles; a fusão deve ser espacial no sistema global da imagem.
