# Fluxo de datasets

As etapas ficam separadas para evitar acoplamento entre descoberta, aumento e treinamento.

```text
Dataset bruto/anotado
        |
        +--> Clusterização (opcional)
        |       DINOv2 + HDBSCAN
        |       ↓
        |    revisão/classificação humana
        |       ↓
        |    exportação do dataset revisado
        |
        +--> divisão train / valid / test
                    ↓
              Data Augmentation
              SOMENTE em train
                    ↓
              dataset YOLO final
                    ↓
                 Treinamento
```

## Dataset Roboflow já revisado

Para o Modelo Geral de detecção, se o Roboflow já contém bounding boxes revisadas e os splits train/valid/test, a clusterização não é necessária.

Use diretamente a página **Data Augmentation**. Ela importa/seleciona um `data.yaml`, copia `valid` e `test` sem alteração e cria novas imagens apenas no `train`.

O dataset aumentado é registrado localmente em `workspace/registry.yaml` e pode ser enviado para `datasets.det_general` em `configs/runtime.yaml` pelo botão da própria página.

## Clusterização

A página **Clusterização** é independente. Ela recebe um dataset YOLO e grava um workspace próprio em `workspace/...`. Workspaces criados pela interface são registrados em `workspace/registry.yaml` para aparecerem novamente no seletor.

Clusterização não dispara augmentation nem treinamento automaticamente.

## Augmentation

A página **Data Augmentation** suporta, de forma selecionável:

- flip horizontal e vertical;
- rotação 90° e rotação livre;
- affine, escala, translação e shear;
- brilho/contraste;
- gamma/exposição;
- hue/saturação/valor;
- Gaussian blur;
- motion blur;
- ruído gaussiano;
- grayscale;
- coarse dropout/cutout;
- mosaic 2x2.

A regra obrigatória é: **valid e test nunca recebem augmentation**.

## Treinamento

A página **Treinamento** usa `configs/runtime.yaml` e `configs/training_experiments.yaml`. O dataset produzido pela augmentation pode ser usado diretamente como `det_general`.
