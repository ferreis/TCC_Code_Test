# Google Colab

O projeto pode rodar no Google Colab usando o notebook:

`notebooks/TCC_Object_Discovery_Colab.ipynb`

## Preparação

1. Abra o notebook no Colab.
2. Em **Runtime > Change runtime type**, escolha **GPU**.
3. Em **Secrets**, crie:
   - `GITHUB_TOKEN`: token com acesso de leitura ao repositório privado `ferreis/TCC_Code_Test`;
   - `ROBOFLOW_API_KEY`: chave da conta Roboflow;
   - `NGROK_AUTHTOKEN`: opcional, necessário apenas para abrir a interface Streamlit por URL pública.
4. Execute as células na ordem.

## Persistência

O notebook monta o Google Drive e cria:

```text
MyDrive/TCC_Code_Test_runtime/
├── data/
├── workspace/
├── exports/
└── weights/
```

Essas pastas são ligadas ao repositório clonado em `/content/TCC_Code_Test` por links simbólicos. Assim, dataset, checkpoints, catálogos, embeddings e exportações continuam no Drive mesmo quando a máquina do Colab é descartada.

## Dependências

Use `requirements-colab.txt`. O arquivo não reinstala `torch` nem `torchvision`, pois o Colab já fornece uma versão do PyTorch integrada ao runtime CUDA.

## Fluxo recomendado

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

O projeto padrão no notebook é:

```text
workspace/bbox_clusters_colab
```

## Interface Streamlit

A última célula do notebook inicia `streamlit run app.py` e cria um túnel ngrok. Para isso, configure `NGROK_AUTHTOKEN` em Secrets. Não coloque tokens diretamente no notebook.

## Observação sobre sessões longas

O Colab pode encerrar sessões por inatividade, limite de uso ou mudança de runtime. A persistência no Drive evita perder os arquivos já gravados, mas um processo em execução é interrompido quando a sessão termina. Para experimentos longos, prefira checkpoints/projetos separados e preserve os resultados em `workspace/`.
