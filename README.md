# TCC Code Test — Object Discovery Lab

Protótipo para **descobrir objetos sem classes pré-definidas**, agrupar objetos visualmente semelhantes e permitir que uma pessoa nomeie os grupos depois.

O pipeline gera ao mesmo tempo:

- **bounding box** de cada proposta;
- **máscara de segmentação**;
- crop do objeto;
- **embedding DINOv2**;
- **cluster HDBSCAN**;
- interface Streamlit para nomear clusters ou corrigir objetos individuais;
- exportação em **YOLO detection** e **YOLO segmentation**;
- data augmentation preservando as máscaras.

## Como funciona

```text
Imagens (Roboflow ou pasta local)
        |
        v
SAM - Segment Anything
  |             |
  |             +--> máscara de segmentação
  +----------------> bounding box
        |
        v
Crop de cada proposta
        |
        v
DINOv2 -> embedding visual
        |
        v
HDBSCAN -> clusters sem classe
        |
        v
Revisão humana
"cluster 3 = baleia", "cluster 8 = barco", ...
        |
        +--> YOLO bounding boxes
        +--> YOLO segmentation
        +--> dataset aumentado
```

**Importante:** o SAM propõe regiões/objetos, mas não sabe que uma região é "baleia", "barco" ou "espuma". O objetivo deste projeto é justamente descobrir e agrupar primeiro e classificar depois.

## 1. Instalação

Python 3.10+ é recomendado.

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se houver GPU NVIDIA, instale a versão do PyTorch adequada à sua versão de CUDA antes de executar o `requirements.txt`.

## 2. Baixar o SAM

O modelo padrão é o `vit_b`, por ser menor e adequado para prototipagem:

```bash
python scripts/download_sam.py
```

Isso cria:

```text
weights/sam_vit_b_01ec64.pth
```

Também é possível testar `--model vit_l` ou `--model vit_h`, exigindo mais VRAM/RAM.

O DINOv2 (`facebook/dinov2-small`) é baixado automaticamente pelo Transformers na primeira execução.

## 3. Dataset do Roboflow

Copie o exemplo:

```bash
cp .env.example .env
```

No Windows:

```powershell
Copy-Item .env.example .env
```

Preencha:

```env
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=...
ROBOFLOW_PROJECT=...
ROBOFLOW_VERSION=1
```

Depois:

```bash
python scripts/download_roboflow.py --format yolov8 --output data/roboflow
```

Para este protótipo, as anotações já existentes no Roboflow não são obrigatórias: o pipeline procura recursivamente pelas imagens dentro da pasta informada.

## 4. Executar a interface

```bash
streamlit run app.py
```

Fluxo da interface:

1. **Descobrir objetos** — SAM cria máscaras e bounding boxes; DINOv2 cria embeddings; HDBSCAN agrupa.
2. **Classificar clusters** — veja crops semelhantes e dê um nome ao cluster. É possível corrigir um objeto individual.
3. **Exportar YOLO** — gera detection e segmentation simultaneamente.
4. **Aumentar dataset** — cria novas imagens e transforma as máscaras junto com a imagem.

## Imagens aéreas e objetos pequenos

Em imagens de drone, um animal pode ocupar uma fração muito pequena da imagem. Nesse caso, habilite `Tile para objetos pequenos` na tela de descoberta.

Comece com:

```text
Tile: 1024 ou 1536
SAM points_per_side: 32
Área mínima: 120 px
```

Se houver muitos falsos objetos, aumente `Área mínima` ou reduza `Máx. objetos/imagem`. Se animais pequenos estiverem sendo perdidos, aumente `points_per_side` e/ou use tiles menores.

## Saída do projeto

```text
workspace/discovery/
├── catalog.json
├── embeddings.npy
├── crops/
└── masks/
```

A máscara é salva apenas dentro do bounding box para evitar armazenar uma imagem binária de resolução completa para cada objeto.

## Exportação YOLO

```text
exports/yolo/
├── images/
├── labels_detection/      # class cx cy w h
├── labels_segmentation/   # class x1 y1 x2 y2 ...
├── classes.txt
└── data.yaml
```

A exportação usa somente objetos que receberam uma classe e que não foram marcados como ignorados.

## Data augmentation

O gerador local cobre os principais efeitos mostrados no Roboflow:

- Flip horizontal e vertical;
- rotação de 90°;
- rotação aleatória;
- shear/affine;
- brilho;
- exposição/gamma;
- blur;
- motion blur;
- ruído;
- saturação e hue;
- grayscale probabilístico;
- cutout/coarse dropout;
- mosaic 2x2;
- efeitos fotométricos **dentro da região do objeto** (equivalentes à ideia de BB Brightness, BB Exposure, BB Blur, BB Noise, BB Motion Blur e BB Camera Gain).

As transformações geométricas são aplicadas à imagem **e às máscaras ao mesmo tempo**. Os bounding boxes são recalculados a partir das máscaras transformadas, evitando boxes fora de sincronia.

Transformações geométricas somente dentro do bounding box, como `BB Rotation` ou `BB Shear`, não são ativadas nesta primeira versão porque podem criar bordas artificiais no animal e alterar a distribuição visual do objeto. Para o TCC, é melhor comparar esse tipo de augmentation separadamente em um experimento controlado.

## Observações importantes para o TCC

1. **SAM não é detector semântico.** Ele encontra regiões candidatas; a semântica é adicionada posteriormente pelo usuário.
2. **Clustering não garante classes biológicas.** Pode separar por cor da água, iluminação, orientação ou escala. Isso é útil para investigar domain shift, mas exige revisão humana.
3. Para cetáceos pequenos em imagens aéreas, o uso de **tiling** é relevante e deve ser avaliado experimentalmente.
4. Não misture imagens aumentadas em validação/teste. Gere augmentations somente para o conjunto de treinamento.
5. O sistema exporta bounding box e segmentation a partir da mesma máscara, permitindo comparar depois YOLO Detection e YOLO Segmentation sobre o mesmo conjunto.

## Próximos experimentos sugeridos

- comparar SAM `vit_b`, `vit_l` e `vit_h`;
- medir quantidade de propostas úteis vs. falsas propostas;
- comparar clustering DINOv2 com embeddings ResNet/ViT;
- medir esforço humano: quantidade de objetos revisados individualmente vs. rotulagem por cluster;
- usar os dados revisados para treinar um YOLO de detecção e um YOLO de segmentação;
- depois aplicar fine-tuning/LoRA e comparar custo computacional e qualidade.
