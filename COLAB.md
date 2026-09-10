# Google Colab

O projeto usa um único notebook:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
```

Ele concentra preparação, clustering/revisão e treinamento YOLO26.

## Preparação

1. Abra o notebook no Colab.
2. Selecione GPU.
3. Configure em Secrets: `GITHUB_TOKEN`, `ROBOFLOW_API_KEY` e, opcionalmente, `NGROK_AUTHTOKEN`.
4. Execute as células na ordem.

## Ambiente Python

Não execute:

```bash
pip install --upgrade pip setuptools wheel
```

O notebook preserva o PyTorch/CUDA do Colab, instala `requirements-colab.txt` e depois instala Tencent/YOLO-Master de forma não-editável e valida `import ultralytics` e `ultralytics.utils.lora` antes de continuar.

## Persistência

```text
MyDrive/TCC_Code_Test_runtime/
├── data/
├── workspace/
├── exports/
└── weights/
```

Essas pastas são ligadas ao clone em `/content/TCC_Code_Test`.

## Preparação dos dados

O dataset amplo de detecção vem do Roboflow. O clustering das boxes revisadas continua disponível via DINOv2 + HDBSCAN e a interface Streamlit pode ser aberta por ngrok.

## Nova nomenclatura de treinamento

A nomenclatura anterior `Base | Base + LoRA | Full` foi removida porque não representava a hipótese do TCC.

Agora usamos:

```text
MODELO GERAL
LoRA ESPECIALISTA 01..05
FULL FINE-TUNE CONTROLE 01..05
SMOKE TEST YOLO26n
```

### Modelo Geral

É treinado no dataset amplo e gera o checkpoint comum:

```text
det_general
    ↓
workspace/training_runs/det_general_yolo26n/weights/best.pt
```

Para segmentação, o equivalente é `seg_general`.

### LoRAs especialistas

Cada LoRA é treinada separadamente e sempre começa do mesmo checkpoint geral:

```text
det_general/best.pt
    ├── det_lora_01 ← dataset pequeno 01
    ├── det_lora_02 ← dataset pequeno 02
    ├── det_lora_03 ← dataset pequeno 03
    ├── det_lora_04 ← dataset pequeno 04
    └── det_lora_05 ← dataset pequeno 05
```

O artefato principal é:

```text
workspace/training_runs/det_lora_01/lora_adapter/
```

Os adapters não são fundidos durante o treinamento.

### Full Fine-Tune de controle

Para cada LoRA N existe um controle Full Fine-Tune N que parte do mesmo `best.pt` geral e usa exatamente o mesmo dataset pequeno N. Assim a comparação LoRA × Full Fine-Tune é justa.

## Execução no notebook

O notebook possui um helper `run_stage()` e cada célula executa **uma única etapa**.

Exemplos conceituais:

```python
run_stage('smoke_det_yolo26n')
run_stage('det_general')
run_stage('det_lora_01')
run_stage('det_fullft_01')
```

Os templates LoRA/Full FT ficam `enabled: false` no YAML versionado para evitar execução acidental. O helper cria uma cópia em `workspace/training_plan_runtime.yaml` e ativa somente o ID escolhido.

## Exportação automática para TCC_Analise

Depois de cada treinamento real concluído pelo notebook, `scripts/train_experiments.py` atualiza automaticamente:

```text
MyDrive/TCC_Code_Test_runtime/exports/
├── TCC_Analise_models/
└── TCC_Analise_models.zip
```

O pacote é organizado para ser extraído/copied diretamente dentro de `TCC_Analise/models/`:

```text
TCC_Analise_models/
├── detection/
│   ├── general/best.pt
│   ├── lora_01/adapter/
│   ├── lora_02/adapter/
│   ├── lora_03/adapter/
│   ├── lora_04/adapter/
│   ├── lora_05/adapter/
│   ├── fullft_01/best.pt
│   ├── fullft_02/best.pt
│   ├── fullft_03/best.pt
│   ├── fullft_04/best.pt
│   └── fullft_05/best.pt
├── segmentation/
│   └── mesma convenção
├── _metadata/
│   ├── experiment_summary.csv
│   └── experiment_summary.json
└── bundle_manifest.json
```

Somente artefatos já treinados são copiados. Se uma LoRA ou Full Fine-Tune ainda não existir, ela aparece como ausente no `bundle_manifest.json` e será incluída automaticamente na próxima atualização do pacote.

Também é possível reconstruir o pacote manualmente sem treinar novamente:

```bash
python scripts/export_analysis_bundle.py
```

O ZIP fica persistente no Google Drive e pode ser baixado pelo painel de arquivos do Drive/Colab.

## YOLO26n

YOLO26n é a escala inicial para validar o pipeline e medir custo na Tesla T4. Depois o Modelo Geral pode ser repetido com `s` ou `m` se houver justificativa experimental e memória suficiente.

## Inferência modular

LoRA não é um segundo modelo autônomo depois do YOLO. Ela é um adapter aplicado internamente ao mesmo modelo-base. Para representar a ideia de especialização em cadeia, serão avaliadas depois:

```text
A) Modelo Geral + adapter escolhido
B) Modelo Geral → ROI candidato → Modelo Geral + LoRA especialista
C) Modelo Geral → múltiplos especialistas → fusão de detecções
```

A estratégia B corresponde ao fluxo de passar primeiro pelo modelo geral e depois por um especialista, mas o segundo estágio é uma nova inferência do modelo-base com o adapter carregado.

O YOLO-Master também possui MoLoRA, porém isso é outro método: os especialistas são treinados dentro de uma arquitetura roteada. Não é o mesmo que treinar cinco LoRAs independentes e juntá-las depois.

## Resultados

Cada etapa grava:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

Para LoRA, o relatório registra também o checkpoint geral de origem, o diretório do adapter, tamanho do adapter e validação executada como `base + adapter`.

Detalhes metodológicos: `TRAINING.md`.
