# Google Colab

O projeto pode rodar no Google Colab usando dois notebooks:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
notebooks/TCC_YOLO26_Experiments_Colab.ipynb
```

O primeiro prepara descoberta, clustering e interface. O segundo é dedicado à bateria controlada de treinamento YOLO26.

## Preparação

1. Abra o notebook no Colab.
2. Em **Runtime > Change runtime type**, escolha **GPU**.
3. Em **Secrets**, crie:
   - `GITHUB_TOKEN`: token com acesso de leitura ao repositório privado `ferreis/TCC_Code_Test`;
   - `ROBOFLOW_API_KEY`: chave da conta Roboflow;
   - `NGROK_AUTHTOKEN`: opcional, necessário apenas para abrir a interface Streamlit por URL pública.
4. Execute as células na ordem.

## Persistência

Os notebooks montam o Google Drive e usam:

```text
MyDrive/TCC_Code_Test_runtime/
├── data/
├── workspace/
├── exports/
└── weights/
```

Essas pastas são ligadas ao repositório clonado em `/content/TCC_Code_Test` por links simbólicos. Assim, dataset, checkpoints, catálogos, embeddings, experimentos e exportações continuam no Drive mesmo quando a máquina do Colab é descartada.

## Dependências

Use `requirements-colab.txt`. O Colab já fornece PyTorch/CUDA, portanto o arquivo não reinstala `torch` nem `torchvision`.

A distribuição oficial `ultralytics` é instalada para YOLO26. Para os experimentos LoRA, o notebook de treinamento instala também o fork Tencent/YOLO-Master, que mantém a API Ultralytics e acrescenta suporte experimental a LoRA.

## Fluxo de descoberta recomendado

Para o dataset já revisado em bounding boxes:

```text
Roboflow YOLO26
      ↓
boxes anotadas
      ↓
crops completos + contexto
      ↓
DINOv2
      ↓
HDBSCAN
      ↓
classificação/revisão humana
```

O projeto padrão é:

```text
workspace/bbox_clusters_colab
```

## Matriz de treinamento

A configuração está em:

```text
configs/training_experiments.yaml
```

Primeira bateria, usando YOLO26n:

```text
Detection:    Base | LoRA | Full fine-tune
Segmentation: Base | LoRA | Full fine-tune
```

Neste TCC:

- `Base`: pesos pré-treinados com backbone congelado (`freeze=11`);
- `LoRA`: adaptação de baixa dimensão com rank 16, alpha 32 e dropout 0.10;
- `Full fine-tune`: todos os pesos liberados (`freeze=0`).

Antes da bateria longa, execute o smoke test de 3 épocas:

```bash
python scripts/train_experiments.py --group smoke_tests --only detect_yolo26n_smoke
```

Depois:

```bash
python scripts/train_experiments.py --only detect_base detect_lora detect_full
```

A segmentação só inicia quando existir:

```text
data/segmentation/data.yaml
```

Os resultados são consolidados automaticamente em:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

O relatório inclui hiperparâmetros, parâmetros totais e treináveis, percentual treinável, parâmetros LoRA, GPU, VRAM de pico, tempo, tamanho do checkpoint, métricas, velocidade e versões do ambiente.

Detalhes metodológicos estão em `TRAINING.md`.

## Interface Streamlit

A última célula do notebook de descoberta inicia `streamlit run app.py` e cria um túnel ngrok. Para isso, configure `NGROK_AUTHTOKEN` em Secrets. Não coloque tokens diretamente no notebook.

## Observação sobre sessões longas

O Colab pode encerrar sessões por inatividade, limite de uso ou mudança de runtime. A persistência no Drive evita perder os arquivos já gravados, mas um processo em execução é interrompido quando a sessão termina.

Os experimentos podem tentar continuar a partir de `last.pt`:

```bash
python scripts/train_experiments.py --only detect_full --resume
```

Para comparações científicas, mantenha os mesmos splits, seed e hiperparâmetros comuns entre Base, LoRA e Full. O conjunto `test` deve ser reservado para a avaliação final e não para selecionar hiperparâmetros.
