# Protocolo de treinamento YOLO26

Este projeto usa uma matriz fixa para comparar custo computacional e desempenho em detecção e segmentação.

## Regimes comparados

Para evitar que "base" e "full fine-tune" sejam o mesmo experimento, os termos são definidos assim:

| Regime | Definição neste TCC |
|---|---|
| Base | pesos YOLO26 pré-treinados, backbone congelado (`freeze=11`), treinamento do restante da rede |
| Base + LoRA | pesos pré-treinados + adaptadores LoRA; pesos-base preservados e adaptação de baixa dimensão |
| Full fine-tune | pesos pré-treinados e todos os pesos treináveis (`freeze=0`) |

A mesma comparação é repetida para detecção e segmentação de instâncias.

## Matriz principal

1. `detect_base` — YOLO26n Detection / Base
2. `detect_lora` — YOLO26n Detection / LoRA
3. `detect_full` — YOLO26n Detection / Full fine-tune
4. `segment_base` — YOLO26n-seg / Base
5. `segment_lora` — YOLO26n-seg / LoRA
6. `segment_full` — YOLO26n-seg / Full fine-tune

A configuração está em:

```text
configs/training_experiments.yaml
```

A escala `n` é usada primeiro para validar o protocolo inteiro com baixo custo. Depois de fechar o protocolo, os melhores regimes podem ser repetidos com `s` ou `m` sem mudar a metodologia.

## Teste rápido YOLO26n

Antes de qualquer bateria longa:

```bash
python scripts/train_experiments.py --group smoke_tests --only detect_yolo26n_smoke
```

O smoke test usa apenas 3 épocas e serve para verificar dataset, CUDA, VRAM, checkpoints e logging.

Quando o dataset de segmentação existir:

```bash
python scripts/train_experiments.py --group smoke_tests --only segment_yolo26n_smoke
```

## Executar detecção

```bash
python scripts/train_experiments.py \
  --only detect_base detect_lora detect_full
```

## Executar segmentação

```bash
python scripts/train_experiments.py \
  --only segment_base segment_lora segment_full
```

Se `data/segmentation/data.yaml` ainda não existir, os experimentos de segmentação são registrados como `missing_data` e pulados.

## LoRA

A distribuição oficial da Ultralytics suporta YOLO26, mas o experimento LoRA deste repositório utiliza o fork **Tencent/YOLO-Master**, que adiciona adaptadores LoRA mantendo a API `from ultralytics import YOLO`.

No Colab:

```bash
python scripts/setup_yolo_master.py --dir /content/YOLO-Master
```

A configuração inicial usada é:

```text
rank (r): 16
alpha: 32
dropout: 0.10
gradient checkpointing: ligado
include head: ligado
```

O commit do código do TCC, caminho/versão do pacote Ultralytics/YOLO-Master e hardware são gravados nos metadados de cada experimento.

## Parâmetros e métricas registrados

Cada treinamento gera `tcc_metadata/experiment.json` e atualiza:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

São registrados, quando disponíveis:

- nome do experimento, tarefa, regime e checkpoint inicial;
- todos os hiperparâmetros resolvidos;
- versão do Python, PyTorch, CUDA e pacote Ultralytics/YOLO-Master;
- GPU e VRAM total;
- commit do repositório TCC;
- parâmetros totais do modelo;
- parâmetros treináveis;
- percentual de parâmetros treináveis;
- parâmetros LoRA treináveis;
- GFLOPs/informações do modelo quando reportados pela biblioteca;
- tempo total de treinamento;
- pico de VRAM alocada e reservada;
- tamanho de `best.pt`;
- métricas de treino/validação;
- métricas finais no split `test` quando o dataset fornece esse split;
- tempo de avaliação e velocidade de inferência reportada.

## Retomar após desconexão do Colab

Os resultados ficam em `workspace/`, que no notebook Colab é ligado ao Google Drive.

Para tentar retomar um experimento com `last.pt`:

```bash
python scripts/train_experiments.py --only detect_full --resume
```

## Regras para comparação

Para os seis experimentos principais devem permanecer iguais: divisão dos dados, seed, resolução, número máximo de épocas, critério de parada, augmentation e conjunto de teste. O que muda deliberadamente é apenas o regime de adaptação dos pesos.

Não usar o conjunto `test` para escolher hiperparâmetros. Ele serve para a comparação final. Para frames provenientes de vídeo, a divisão deve preferencialmente ocorrer por vídeo ou blocos temporais, evitando que frames quase idênticos apareçam em treino e teste.
