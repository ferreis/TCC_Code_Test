# Google Colab

O projeto usa um único notebook principal no Google Colab:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
```

Esse notebook concentra descoberta, clustering, interface Streamlit e a matriz controlada de treinamento YOLO26.

## Preparação

1. Abra o notebook no Colab.
2. Em **Runtime > Change runtime type**, escolha **GPU**.
3. Em **Secrets**, crie:
   - `GITHUB_TOKEN`: token com acesso de leitura ao repositório privado `ferreis/TCC_Code_Test`;
   - `ROBOFLOW_API_KEY`: chave da conta Roboflow;
   - `NGROK_AUTHTOKEN`: opcional, necessário apenas para abrir a interface Streamlit por URL pública.
4. Execute as células na ordem.

## Importante: não atualizar pip/setuptools/wheel

O notebook foi ajustado para evitar o aviso do Colab:

```text
WARNING: The following packages were previously imported in this runtime: [_distutils_hack]
You must restart the runtime in order to use newly installed versions.
```

Por isso, **não execute** no Colab:

```bash
pip install --upgrade pip setuptools wheel
```

O runtime já fornece essas ferramentas. O notebook instala apenas `requirements-colab.txt`, sem atualizar as ferramentas-base.

Também foi alterada a ordem do setup: o YOLO-Master é instalado **antes do primeiro `import ultralytics`**. Isso evita instalar/substituir o backend depois que o módulo já foi carregado e reduz a necessidade de reiniciar a sessão.

A ordem inicial correta é:

```text
verificar GPU sem importar torch/ultralytics
        ↓
montar Drive
        ↓
clonar/atualizar GitHub
        ↓
instalar requirements-colab.txt
        ↓
instalar YOLO-Master
        ↓
importar e validar PyTorch/CUDA/Ultralytics/LoRA
```

## Persistência

O notebook monta o Google Drive e usa:

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

A distribuição `ultralytics` é instalada como dependência base. Em seguida, antes do primeiro import, o notebook instala o fork Tencent/YOLO-Master em modo editável. Os experimentos Base, LoRA e Full da sessão passam então a usar a mesma implementação.

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

A interface Streamlit pode ser aberta pelo próprio notebook via ngrok para upload de imagem/vídeo, classificação de clusters e revisão.

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

## Uso de uma única GPU

O notebook foi unificado justamente para usar a mesma sessão e a mesma GPU. Antes de iniciar treinamentos longos, existe uma célula para encerrar Streamlit/ngrok e liberar a VRAM.

## Observação sobre sessões longas

O Colab pode encerrar sessões por inatividade, limite de uso ou mudança de runtime. A persistência no Drive evita perder os arquivos já gravados, mas um processo em execução é interrompido quando a sessão termina.

Os experimentos podem tentar continuar a partir de `last.pt`:

```bash
python scripts/train_experiments.py --only detect_full --resume
```

Para comparações científicas, mantenha os mesmos splits, seed e hiperparâmetros comuns entre Base, LoRA e Full. O conjunto `test` deve ser reservado para a avaliação final e não para selecionar hiperparâmetros.
