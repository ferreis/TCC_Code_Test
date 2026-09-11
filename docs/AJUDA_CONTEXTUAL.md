# Ajuda contextual nas interfaces

As definições ficam em `technical_terms.json`. `contextual_help.py` oferece ajuda
nativa nos campos e métricas, ícones ⓘ nos textos e cabeçalhos de tabelas com tooltip.
Nos textos, passe o mouse ou clique no ⓘ; também é possível usar Tab e Enter.
O painel de treinamento tem um popover “ⓘ Termos do log”, preservando o log original.
A visualização JSON mantém os dados originais e oferece “ⓘ Campos deste resultado”.

## Telas atendidas

| Projeto / tela | Termos e controles com ajuda |
| --- | --- |
| TCC_Code_Test — Upload e descoberta SAM | SAM, points_per_side, points_per_batch, crop_n_layers, checkpoint, frames, FPS, amostragem, máscaras, área mínima, objetos, tiles/auto-tile, GPU, CPU, VRAM, fallback |
| TCC_Code_Test — Clusterização independente e por bounding boxes | dataset, data.yaml, split, train, validation, test, workspace, bounding box/box, crop, contexto da box, DINOv2, embeddings, HDBSCAN, clusters, mínimo por cluster, batch DINOv2 |
| TCC_Code_Test — Classificação, revisão e exportação | classes, objetos, clusters, pendentes, classificados, ignorados, máscaras, segmentação, JSON, YOLO |
| TCC_Code_Test — Augmentation (app e página independente) | dataset, train/validation/test, split, augmentation agressiva, cópias, tamanho, seed, mosaic, flip horizontal/vertical, rotação, affine/shear/escala, brilho/contraste, gamma/exposição, hue/saturação/HSV, Gaussian blur, motion blur, ruído gaussiano, grayscale, cutout/coarse dropout, boxes, ETA |
| TCC_Code_Test — Treinamento e resumo | perfil de hardware, etapas, modelo, YOLO26, detecção/segmentação, epochs, imgsz, batch, workers, device, cache, AMP, determinístico, patience, seed, save_period, close mosaic, data.yaml, smoke test, LoRA, rank, alpha, dropout, head, gradient checkpointing, checkpoint, GPU/VRAM/GPU-Util |
| TCC_Code_Test — Termos do log e colunas do resumo | learning rate/lr0/lrf, loss e componentes, precision, recall, mAP50, mAP50-95, checkpoint, VRAM, ETA; colunas reconhecidas recebem ajuda pelo nome |
| TCC_Code_analisador — Inferência visual (imagem, vídeo e webcam) | modelo, checkpoint, adapter/LoRA, confidence, IoU/NMS, imgsz, device, frames, inferência, detecções, classes, confidence e coordenadas x1/y1/x2/y2 nas tabelas |
| TCC_Code_analisador — Revisão de falsos positivos | execução, imagens processadas, detecções positivas, revisadas, falsos/verdadeiros positivos, frame_index, detection_uid, class_name, confidence, coordenadas, review_note, validação da revisão |
| TCC_Code_analisador — Resultados, comparação e JSON | precision, recall, mAP50, mAP50-95, sufixos box/(B) e mask/(M), score_primary, score_map50, loss, split, parâmetros, tempos speed_ms, VRAM, checkpoint, contagens da revisão, false_positive_rate_over_reviewed |

As tabelas só exibem ajuda para colunas presentes; nenhum valor ou chave de dados é
renomeado. A taxa de falsos positivos da revisão é explicada como FP/revisadas,
não como FP/(FP+TN). A validação humana é diferenciada do split de validação.

## Manutenção

Os projetos são independentes: cada um contém sua cópia idêntica do catálogo e do
módulo de apresentação, sem depender do caminho do outro projeto. Ao atualizar
uma definição comum, sincronize `technical_terms.json` e `contextual_help.py` entre
os dois repositórios. Não copie definições para páginas.

- Campos/métricas: `help=help_for("Epochs")` (o rótulo, valor e chave não mudam).
- Texto curto: `help_text("SAM → DINOv2 → HDBSCAN", caption=True)`.
- Tabelas: `help_dataframe(df)` ou `help_editor(df)`; as colunas usam ajuda nativa.
- JSON: `help_json(payload)`.
- Novos termos: acrescente uma entrada com `aliases` e `text` ao catálogo.

Teste: `python -m unittest discover -s tests -v` no ambiente de cada projeto.

## Catálogo central

Os termos abaixo também cobrem nomes técnicos que podem aparecer nas colunas
exportadas, sem criar controles novos para parâmetros que não existem na tela.

- **Affine** (`affine`)
- **Augmentation agressiva** (`aggressive`)
- **Alpha** (`alpha`)
- **AMP** (`amp`)
- **Augmentation** (`augmentation`)
- **Batch** (`batch`)
- **Gaussian blur** (`blur`)
- **bounding boxes** (`box`)
- **Brilho e contraste** (`brightness`)
- **Cache** (`cache`)
- **Checkpoint** (`checkpoint`)
- **checkpoint_sha256** (`checkpoint_hash`)
- **checkpoint_size_mb** (`checkpoint_size`)
- **classe** (`class`)
- **Close mosaic** (`close_mosaic`)
- **cluster** (`cluster`)
- **Confidence** (`confidence`)
- **Contexto ao redor da box** (`context_scale`)
- **x1** (`coordinates`)
- **Cópias por imagem** (`copies`)
- **CPU** (`cpu`)
- **crop** (`crop`)
- **SAM crop_n_layers** (`crop_layers`)
- **Cutout** (`cutout`)
- **data.yaml** (`data_yaml`)
- **Dataset** (`dataset`)
- **detecção** (`detection`)
- **detection_uid** (`detection_uid`)
- **Determinístico** (`deterministic`)
- **Device** (`device`)
- **Batch DINOv2** (`dino_batch`)
- **DINOv2** (`dinov2`)
- **Dropout** (`dropout`)
- **elapsed_seconds** (`duration`)
- **embedding** (`embedding`)
- **Epoch** (`epoch`)
- **ETA** (`eta`)
- **Fallback** (`fallback`)
- **Full Fine-Tune** (`finetune`)
- **Flip** (`flip`)
- **Falsos positivos** (`fp`)
- **false_positive_rate_over_reviewed** (`fp_rate`)
- **FPS** (`fps`)
- **frame** (`frame`)
- **frame_index** (`frame_index`)
- **freeze** (`freeze`)
- **Gamma** (`gamma`)
- **GPU** (`gpu`)
- **Gradient checkpointing** (`gradient_checkpointing`)
- **Grayscale** (`grayscale`)
- **Perfil de hardware** (`hardware`)
- **HDBSCAN** (`hdbscan`)
- **head** (`head`)
- **Hue** (`hsv`)
- **Imagens processadas** (`images_count`)
- **imgsz** (`imgsz`)
- **Inferência** (`inference`)
- **IoU** (`iou`)
- **JSON** (`json`)
- **speed_ms** (`latency`)
- **learning rate** (`learning_rate`)
- **LoRA** (`lora`)
- **Loss** (`loss`)
- **mAP** (`map`)
- **mAP50** (`map50`)
- **mAP50-95** (`map50_95`)
- **máscara** (`mask`)
- **Área mínima da máscara (px)** (`min_area`)
- **Mínimo por cluster** (`min_cluster`)
- **Modelo** (`model`)
- **momentum** (`momentum`)
- **Mosaicos** (`mosaic`)
- **Motion blur** (`motion_blur`)
- **IoU/NMS** (`nms`)
- **Ruído gaussiano** (`noise`)
- **Objeto** (`objects`)
- **optimizer** (`optimizer`)
- **parameters** (`parameters`)
- **Patience** (`patience`)
- **SAM points_per_batch** (`points_batch`)
- **SAM points_per_side** (`points_side`)
- **positive_detections_before_review** (`positive_count`)
- **Precision** (`precision`)
- **pretrained** (`pretrained`)
- **Rank (r)** (`rank`)
- **Recall** (`recall`)
- **review_note** (`review_note`)
- **Pendentes** (`review_status`)
- **Validação da revisão** (`review_validation`)
- **Revisadas** (`reviewed`)
- **rotação** (`rotate`)
- **Execução** (`run`)
- **SAM** (`sam`)
- **Salvar a cada N épocas** (`save_period`)
- **score_primary** (`score_primary`)
- **Seed** (`seed`)
- **segmentação** (`segmentation`)
- **Smoke** (`smoke`)
- **source_name** (`source`)
- **Split** (`split`)
- **Etapa** (`stage`)
- **Test** (`test`)
- **Auto-tile** (`tile`)
- **Ativar auto-tile acima de** (`tile_trigger`)
- **Verdadeiros positivos** (`tp`)
- **Train** (`train`)
- **Validation** (`validation`)
- **VRAM** (`vram`)
- **weight_decay** (`weight_decay`)
- **Workers** (`workers`)
- **Workspace** (`workspace`)
- **YOLO** (`yolo`)
