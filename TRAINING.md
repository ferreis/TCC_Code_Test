# Protocolo de treinamento YOLO26

O projeto usa **um Modelo Geral treinado em dataset amplo + múltiplos adapters LoRA especialistas treinados separadamente em datasets pequenos**, com controles Full Fine-Tune equivalentes.

## Fonte única de configuração

Os parâmetros editáveis ficam em:

```text
configs/runtime.yaml
```

Esse arquivo é compartilhado por três entradas:

```text
Google Colab ─┐
Streamlit ────┼──> scripts/train_experiments.py
Terminal ─────┘              ↓
                    tcc_discovery/training_experiments.py
```

`configs/training_experiments.yaml` não é mais a fonte dos hiperparâmetros. Ele define apenas a topologia das etapas: IDs, regimes, relações `base_from` e nomes de execução.

O runtime possui dois perfis iniciais:

```text
local  → perfil conservador para GPU ~6 GB
colab  → perfil para Tesla T4 ~15 GB + staging do dataset em /content
```

Batch, workers, cache e staging podem diferir por perfil; epochs, resolução, seed, LoRA e demais parâmetros científicos permanecem centralizados no mesmo arquivo.

## Terminologia

| Nome | Definição |
|---|---|
| **Modelo Geral** | YOLO26 treinado no dataset amplo; produz o `best.pt` raiz |
| **LoRA Especialista N** | adapter treinado separadamente sobre o checkpoint do Modelo Geral usando dataset pequeno N |
| **Full Fine-Tune Controle N** | cópia do mesmo checkpoint geral com todos os pesos liberados no mesmo dataset pequeno N |
| **Smoke Test YOLO26n** | teste curto de ambiente; não entra na comparação final |

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

Cada LoRA é independente. `LoRA 02` não continua de `LoRA 01`.

## IDs

```text
smoke_det_yolo26n
smoke_seg_yolo26n

det_general
seg_general

det_lora_01 ... det_lora_05
seg_lora_01 ... seg_lora_05

det_fullft_01 ... det_fullft_05
seg_fullft_01 ... seg_fullft_05
```

Quando `--only` é usado, o runtime ativa explicitamente apenas a etapa selecionada, mesmo que o template esteja `enabled: false` para impedir disparos acidentais em lote.

## Treinamento pelo terminal

Local:

```bash
python scripts/train_experiments.py --profile local --only smoke_det_yolo26n
python scripts/train_experiments.py --profile local --only det_general
python scripts/train_experiments.py --profile local --only det_lora_01
python scripts/train_experiments.py --profile local --only det_fullft_01
```

Colab usa os mesmos comandos com:

```text
--profile colab
```

Listar sem treinar:

```bash
python scripts/train_experiments.py --profile local --group all --list
```

Dry-run:

```bash
python scripts/train_experiments.py --profile local --only det_general --dry-run
```

Retomar:

```bash
python scripts/train_experiments.py --profile local --only det_general --resume
```

## Frontend de treinamento

Execute:

```bash
streamlit run app.py
```

O diretório `pages/` adiciona a página **Treinamento** ao Streamlit. Ela permite selecionar perfil e etapa, editar/salvar `configs/runtime.yaml`, visualizar GPU/VRAM/GPU-Util, iniciar o treinamento em processo separado, acompanhar o log, atualizar o status, parar a execução e consultar os resultados já registrados.

Para uma etapa LoRA, a página verifica `ultralytics.utils.lora`. Se necessário, pode executar `scripts/setup_yolo_master.py` no mesmo ambiente virtual. O treinamento em si continua sendo feito pela mesma CLI; o frontend não possui uma segunda implementação do algoritmo.

## LoRA

O backend é Tencent/YOLO-Master. O adapter é salvo separadamente com `save_lora_only()` e, na avaliação, é recarregado sobre o mesmo checkpoint geral.

Valores iniciais estão em `configs/runtime.yaml`:

```text
rank: 16
alpha: 32
dropout: 0.10
gradient checkpointing: ligado
include head: ligado
```

O projeto não faz `merge_lora()` durante o protocolo principal.

## Comparação LoRA × Full Fine-Tune

Para o especialista N:

```text
mesmo checkpoint geral
+ mesmo dataset pequeno N
+ mesmos hiperparâmetros científicos
```

Muda apenas o regime:

```text
LoRA N           → pesos-base preservados + adapter treinável
Full Fine-Tune N → todos os pesos treináveis
```

Medimos qualidade, tempo, VRAM, parâmetros treináveis, tamanho do artefato e velocidade.

## Datasets e staging

Os caminhos de todos os `data.yaml` ficam em `configs/runtime.yaml`.

No perfil `colab`, `use_local_dataset_cache: true` faz a CLI copiar apenas o dataset da etapa atual para SSD local em `/content/TCC_training_data` antes do treino. Os checkpoints/resultados continuam no diretório persistido ligado ao Google Drive.

No perfil `local`, o staging começa desativado porque o dataset já está em disco local.

## Artefatos

```text
Modelo Geral:
workspace/training_runs/det_general_yolo26n/weights/best.pt

LoRA:
workspace/training_runs/det_lora_01/lora_adapter/

Full Fine-Tune:
workspace/training_runs/det_fullft_01/weights/best.pt
```

Cada execução atualiza:

```text
workspace/training_runs/experiment_summary.csv
workspace/training_runs/experiment_summary.json
```

Após um treino real, `scripts/train_experiments.py` também atualiza automaticamente:

```text
exports/TCC_Analise_models/
exports/TCC_Analise_models.zip
```

## Estratégias de inferência posteriores

```text
A) Modelo Geral + adapter selecionado
B) Modelo Geral → ROI candidato → Modelo Geral + LoRA especialista
C) Modelo Geral → vários especialistas → fusão espacial das detecções
```

A estratégia B representa melhor a hipótese de primeiro localizar com conhecimento geral e depois reavaliar a região com um especialista.

## Regra de avaliação

Não usar `test` para selecionar hiperparâmetros. Para vídeos, separar treino/validação/teste por vídeo ou blocos temporais para evitar vazamento por frames quase idênticos.
