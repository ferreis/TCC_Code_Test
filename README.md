# TCC Code Test — Object Discovery Lab

Protótipo para descoberta, agrupamento, revisão e treinamento de modelos em imagens aéreas de cetáceos.

O projeto possui fluxos complementares:

1. **Baseline totalmente automático:** imagem → SAM → propostas/máscaras → DINOv2 → HDBSCAN.
2. **Fluxo recomendado para o dataset já anotado:** bounding boxes Roboflow → crops completos → DINOv2 → HDBSCAN → revisão humana → SAM guiado pela box (próxima etapa) → segmentação.
3. **Matriz experimental YOLO26:** Detection e Segmentation em três regimes comparáveis: Base, LoRA e Full fine-tune.

A segunda abordagem evita que um cetáceo já anotado seja cortado em partes apenas porque atravessa a borda de um tile.

## Instalação

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
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## SAM

O protótipo usa SAM ViT-B:

```bash
python scripts/download_sam.py
```

Checkpoint padrão:

```text
weights/sam_vit_b_01ec64.pth
```

## Interface

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True streamlit run app.py
```

### 0. Upload e clusterizar

Permite enviar uma imagem ou vídeo, extrair todos os frames ou amostrar por intervalo e executar SAM → DINOv2 → HDBSCAN. O vídeo preserva a referência ao frame original.

### 1. Descobrir com SAM

Baseline sem usar as anotações existentes. Para imagens grandes, o sistema usa tiles adaptativos e devolve as coordenadas para o sistema da imagem original.

Esse fluxo é útil como experimento, mas um objeto que atravesse exatamente uma borda de tile ainda pode aparecer em fragmentos. Por isso ele não é o fluxo principal para as bounding boxes já revisadas.

### 2. Clusterizar bounding boxes

Fluxo recomendado para o dataset atual.

Use um projeto separado, por exemplo:

```text
workspace/bbox_clusters_v1
```

Pasta do split:

```text
data/roboflow/train
```

Cada anotação YOLO é convertida de coordenadas normalizadas para pixels, o cetáceo é recortado inteiro com contexto ao redor e o crop é enviado ao DINOv2. O HDBSCAN agrupa os embeddings visualmente semelhantes.

Configuração inicial recomendada:

```text
Contexto ao redor da box: 1.8
Crop mínimo: 64 px
Mínimo por cluster: 6
Batch DINOv2: 4
Fallback CPU: ligado
```

Para objetos de poucos pixels, o crop mínimo evita enviar ao encoder uma região extremamente pequena. Isso não cria detalhe inexistente na imagem, apenas preserva um contexto controlado para o embedding.

### 3. Classificar pendentes

A fila mostra apenas objetos ainda sem decisão.

Ao aplicar uma classe ao grupo, ele sai da fila de pendentes. Ao ignorar, ele vai para a fila de ignorados.

### 4. Revisar classificados

Mostra objetos e clusters que já receberam classe. É possível:

- alterar a classe do grupo;
- reabrir o grupo como pendente;
- mudar apenas um objeto;
- marcar um falso positivo como ignorado.

### 5. Revisar ignorados

Permite restaurar um grupo ou objeto descartado sem apagar informação do `catalog.json`.

## Estado do projeto

Projeto SAM:

```text
workspace/discovery/
├── catalog.json
├── embeddings.npy
├── crops/
└── masks/
```

Projeto de bounding boxes:

```text
workspace/bbox_clusters_v1/
├── catalog.json
├── embeddings.npy
└── crops/
```

No projeto de boxes, `mask_path` começa vazio porque a segmentação ainda não foi gerada. A próxima etapa é usar a própria box como prompt para o SAM, em um ROI centrado no objeto, evitando cortar o cetáceo por tiles.

## Matriz de treinamento YOLO26

A configuração experimental está em:

```text
configs/training_experiments.yaml
```

A primeira bateria usa YOLO26n e YOLO26n-seg:

```text
Detection
├── Base: backbone congelado
├── LoRA
└── Full fine-tune

Segmentation
├── Base: backbone congelado
├── LoRA
└── Full fine-tune
```

Antes de iniciar os experimentos longos:

```bash
python scripts/train_experiments.py --group smoke_tests --only detect_yolo26n_smoke
```

Depois, para detecção:

```bash
python scripts/train_experiments.py --only detect_base detect_lora detect_full
```

Para LoRA, utilize o fork YOLO-Master:

```bash
python scripts/setup_yolo_master.py --dir /content/YOLO-Master
```

Os resultados de todos os experimentos são consolidados automaticamente em:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

O relatório guarda hiperparâmetros, parâmetros totais/treináveis, parâmetros LoRA, hardware, pico de VRAM, tempo de treino, tamanho dos checkpoints, métricas e velocidade de avaliação.

Detalhes estão em `TRAINING.md`.

## Arquitetura alvo do TCC

```text
Bounding boxes revisadas no Roboflow
              |
              +--> DINOv2 --> HDBSCAN --> classificação/revisão de clusters
              |
              +--> treino YOLO26 detector
              |         |
              |         +--> procurar possíveis objetos não anotados
              |         +--> inferência em tiles + fusão no sistema global
              |
              +--> SAM guiado pela box --> máscaras --> dataset de segmentação
                        |
                        +--> YOLO26-seg Base / LoRA / Full fine-tune
```

Para auditoria de anotações, a ideia é treinar o detector em partições sem incluir a imagem que está sendo auditada (out-of-fold) e comparar as predições com as boxes humanas por IoU. Predições fortes sem correspondência viram candidatos para revisão humana, não rótulos automáticos definitivos.

## Imagens de vídeo e objetos quase pontuais

Quando o animal ocupa apenas um ou poucos pixels, nenhuma segmentação consegue recuperar um contorno que não existe na resolução original. Nesses frames, detecção e consistência temporal podem ser mais confiáveis que uma máscara detalhada.

Para vídeos, o plano é combinar detecção por frame com rastreamento/consistência temporal. Um candidato que aparece de forma coerente em quadros consecutivos merece prioridade maior na auditoria do que um único pixel branco isolado em um só frame.

## Google Colab

Existem dois notebooks:

```text
notebooks/TCC_Object_Discovery_Colab.ipynb
notebooks/TCC_YOLO26_Experiments_Colab.ipynb
```

Eles utilizam:

```text
MyDrive/TCC_Code_Test_runtime/
├── data/
├── workspace/
├── exports/
└── weights/
```

Use `requirements-colab.txt`, que evita reinstalar `torch`/`torchvision` sobre o runtime CUDA já fornecido pelo Colab.

No **Colab > Secrets**, configure:

```text
GITHUB_TOKEN
ROBOFLOW_API_KEY
NGROK_AUTHTOKEN   # opcional, só para abrir o Streamlit
```

Instruções detalhadas estão em `COLAB.md`.

## Regras experimentais

- Não misturar `train`, `valid` e `test` na criação do modelo final.
- Se frames consecutivos vierem do mesmo vídeo, preferir divisão por vídeo ou por blocos temporais para reduzir vazamento entre treino e teste.
- Manter o baseline SAM automático para comparação, mesmo que tenha desempenho ruim.
- Medir cobertura, falsos candidatos, tempo computacional e esforço humano.
- Aplicar augmentation apenas no conjunto de treinamento.
- Na comparação Base × LoRA × Full, manter splits, seed, resolução, augmentation e protocolo comuns iguais; mudar apenas o regime de adaptação.
