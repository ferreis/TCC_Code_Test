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

## Regra do ambiente Python

No Colab, não execute:

```bash
pip install --upgrade pip setuptools wheel
```

O notebook não atualiza essas ferramentas. Também não instala primeiro o `ultralytics` oficial para depois substituí-lo.

O arquivo `requirements-colab.txt` instala apenas as dependências necessárias, preservando o PyTorch/CUDA fornecido pelo próprio Colab. Em seguida, `scripts/setup_yolo_master.py` clona o Tencent/YOLO-Master e instala o pacote `ultralytics` diretamente desse repositório.

A instalação padrão do YOLO-Master é **não-editável**. Isso é intencional: instalações editáveis podem depender de arquivos `.pth` processados pelo Python na inicialização e, em notebooks, isso pode tornar o pacote invisível até um reinício de kernel.

O setup usa o mesmo `sys.executable` do notebook e, ao final, abre um segundo processo Python com esse mesmo interpretador para validar:

```text
import ultralytics
ultralytics.utils.lora disponível
caminho do pacote instalado
```

Se essa verificação falhar, a célula de instalação falha imediatamente. O notebook não deve avançar para treino nesse caso.

A ordem inicial é:

```text
verificar GPU sem importar torch/ultralytics
        ↓
montar Drive
        ↓
clonar/atualizar GitHub
        ↓
instalar requirements-colab.txt
        ↓
instalar YOLO-Master não-editável
        ↓
validar import em processo Python novo
        ↓
validar novamente dentro do kernel
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

`requirements-colab.txt` não reinstala `torch` nem `torchvision` e não contém o pacote oficial `ultralytics`. As dependências do YOLO-Master, incluindo `peft`, `ultralytics-thop`, `polars` e `nvidia-ml-py`, são preparadas separadamente. Depois o fork é instalado com `--no-deps`, evitando que sua instalação tente trocar o PyTorch CUDA do Colab.

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

O notebook foi unificado para usar a mesma sessão e a mesma GPU. Antes dos treinamentos longos, existe uma célula para encerrar Streamlit/ngrok e liberar VRAM.

## Observação sobre sessões longas

O Colab pode encerrar sessões por inatividade, limite de uso ou mudança de runtime. A persistência no Drive evita perder os arquivos já gravados, mas um processo em execução é interrompido quando a sessão termina.

Os experimentos podem tentar continuar a partir de `last.pt`:

```bash
python scripts/train_experiments.py --only detect_full --resume
```

Para comparações científicas, mantenha os mesmos splits, seed e hiperparâmetros comuns entre Base, LoRA e Full. O conjunto `test` deve ser reservado para a avaliação final e não para selecionar hiperparâmetros.
