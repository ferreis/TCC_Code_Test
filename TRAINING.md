# Protocolo de treinamento YOLO26

O treinamento foi reorganizado para refletir a hipótese real do TCC: **um modelo geral treinado em dataset amplo + múltiplos adapters LoRA especialistas treinados separadamente em datasets pequenos**.

## Terminologia

Não usamos mais `Base`, `Base + LoRA` e `Full` como três treinamentos paralelos sobre o mesmo dataset amplo.

| Nome | Definição |
|---|---|
| **Modelo Geral** | modelo YOLO26 treinado no dataset amplo; produz o checkpoint raiz `best.pt` |
| **LoRA Especialista N** | adapter pequeno treinado separadamente sobre o checkpoint do Modelo Geral, usando um dataset pequeno N |
| **Full Fine-Tune Controle N** | cópia do mesmo checkpoint geral com todos os pesos liberados, treinada no mesmo dataset pequeno N da LoRA N |
| **Smoke Test YOLO26n** | teste curto de ambiente; não entra na comparação científica final |

## Relação entre o modelo geral e as LoRAs

```text
Dataset amplo
    ↓
MODELO GERAL
    ↓
best.pt
    ├── LoRA 01 ← dataset pequeno 01
    ├── LoRA 02 ← dataset pequeno 02
    ├── LoRA 03 ← dataset pequeno 03
    ├── LoRA 04 ← dataset pequeno 04
    └── LoRA 05 ← dataset pequeno 05
```

Cada LoRA é treinada **independentemente**. O treinamento de `LoRA 02` não continua a partir de `LoRA 01`; ambas partem do mesmo `best.pt` do Modelo Geral.

No YOLO-Master, o adapter é salvo separadamente por `save_lora_only()`. O projeto não executa `merge_lora()` durante o protocolo de treinamento, porque queremos preservar a modularidade e medir o custo de cada adapter.

## Comparação com Full Fine-Tune

Para avaliar se a LoRA compensa, o controle correto é:

```text
mesmo Modelo Geral
+ mesmo dataset pequeno N
+ mesmos hiperparâmetros comuns
```

A única diferença deliberada deve ser:

```text
LoRA N              → pesos-base preservados + adapter treinável
Full Fine-Tune N    → todos os pesos do modelo liberados
```

Assim podemos comparar qualidade, tempo, VRAM, parâmetros treináveis e tamanho do artefato.

## Configuração

O plano está em:

```text
configs/training_experiments.yaml
```

Os IDs principais são:

```text
det_general
seg_general

det_lora_01 ... det_lora_05
seg_lora_01 ... seg_lora_05

det_fullft_01 ... det_fullft_05
seg_fullft_01 ... seg_fullft_05

smoke_det_yolo26n
smoke_seg_yolo26n
```

Os templates especialistas ficam `enabled: false` no arquivo versionado para evitar execução acidental em lote. O notebook cria uma cópia de runtime e ativa somente a etapa escolhida.

## Ordem de treinamento

### Detecção

```text
1. smoke_det_yolo26n
2. det_general
3. det_lora_01
4. det_fullft_01
5. det_lora_02
6. det_fullft_02
...
```

Não é obrigatório treinar cinco LoRAs; `01..05` são slots para experimentos especialistas.

### Segmentação

A mesma lógica é usada depois que `data/segmentation/data.yaml` existir:

```text
seg_general
    ↓
seg_lora_01 ... seg_lora_05
    ↓
seg_fullft_01 ... seg_fullft_05
```

## YOLO26n

YOLO26n é usado inicialmente porque permite validar o protocolo na Tesla T4 com menor custo. Isso não significa que o Modelo Geral final precise permanecer `n`; depois de estabilizar o pipeline podemos repetir o Modelo Geral com `s` ou `m`, mantendo a mesma metodologia.

## LoRA

O backend é o **Tencent/YOLO-Master**, que expõe `save_lora_only()`, `load_lora()` e `merge_lora()`.

Configuração inicial dos adapters:

```text
rank (r): 16
alpha: 32
dropout: 0.10
gradient checkpointing: ligado
include head: ligado
```

Esses valores são ponto de partida e devem ser registrados em cada execução.

## Artefatos

Modelo Geral:

```text
workspace/training_runs/det_general_yolo26n/weights/best.pt
```

Exemplo de LoRA:

```text
workspace/training_runs/det_lora_01/lora_adapter/
```

Exemplo de Full Fine-Tune:

```text
workspace/training_runs/det_fullft_01/weights/best.pt
```

Para LoRA, o artefato principal registrado é o adapter. A validação final do experimento LoRA recarrega explicitamente:

```text
Modelo Geral best.pt + lora_adapter/
```

em vez de depender de um checkpoint fundido.

## Estratégias de inferência a testar depois

A LoRA não é um detector completo que recebe diretamente a saída do Modelo Geral. Ela modifica internamente camadas do mesmo modelo-base. Para reproduzir a ideia de especialização em duas etapas, testaremos três arquiteturas:

```text
A) Geral + adapter selecionado
B) Geral → ROI candidato → Geral + LoRA especialista
C) Geral → vários especialistas → fusão espacial das detecções
```

A estratégia B é a que mais se aproxima da ideia de "passar primeiro pelo modelo geral e depois pelo vínculo especialista". O segundo estágio executa novamente o modelo-base com o adapter carregado sobre a região candidata.

Treinar várias LoRAs separadas e simplesmente somá-las depois não é equivalente a esse fluxo. O YOLO-Master também oferece MoLoRA, mas MoLoRA é outro método, com especialistas treinados dentro de uma arquitetura roteada; não deve ser confundido com cinco adapters independentes deste experimento.

## Métricas registradas

Cada etapa atualiza:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

São registrados, quando disponíveis: hiperparâmetros, modelo-base de origem, parâmetros totais e treináveis, parâmetros LoRA, tempo de treino, GPU, pico de VRAM, tamanho de `best.pt`, tamanho do adapter LoRA, métricas de box/máscara, velocidade de inferência, versões do ambiente e commit do projeto.

## Regra de avaliação

Não usar `test` para selecionar hiperparâmetros. Para vídeos, dividir treino/validação/teste por vídeo ou por blocos temporais, evitando frames quase idênticos em conjuntos diferentes.
