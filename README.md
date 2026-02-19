# Sistema Multimodal de Triagem — Saúde da Mulher

Sistema de análise multimodal (vídeo + áudio) para detecção precoce de sinais de **depressão**, **violência doméstica** e **problemas de saúde**, voltado para o contexto da saúde feminina.

> **Triagem automatizada** — NÃO substitui avaliação profissional.

## Objetivos Atendidos (POS TECH — Tech Challenge Fase 4)

| # | Objetivo | Como é atendido |
|---|----------|-----------------|
| 2 | Identificar sinais de violência doméstica | Detecção de hematomas/marcas via YOLOv8 + análise HSV |
| 3 | Monitorar bem-estar psicológico feminino | Expressões faciais (MediaPipe) + análise linguística (Whisper) |
| 5 | Detecção de anomalias para monitoramento preventivo | Pipeline automatizado com sistema de alertas por nível de risco |

## Arquitetura / Fluxo Multimodal

```
Vídeo (MP4)
  │
  ├─► YOLOv8 (detecção de pessoa no frame)
  │     └─► MediaPipe FaceMesh (landmarks faciais)
  │           ├─► Análise de expressão (eye aspect ratio, mouth ratio)
  │           └─► Detecção de anomalias na pele (HSV: hematomas, marcas)
  │
  ├─► FFmpeg (extração de áudio)
  │     └─► Whisper base (transcrição local PT-BR)
  │           ├─► Análise linguística (keywords depressão/violência/pós-parto)
  │           └─► Detecção de hesitações e padrões negativos
  │
  └─► librosa (features vocais: pitch, energia, pausas)
        └─► Indicadores de tristeza/fadiga vocal

  Fusão Multimodal: score_visual × 0.4 + score_audio × 0.6
  └─► Relatório integrado com nível de risco e recomendações
```

## Modelos Utilizados

| Modelo | Tipo de dado | Função |
|--------|-------------|--------|
| **YOLOv8n** (ultralytics) | Vídeo | Detecção de pessoas no frame (pré-filtro) |
| **MediaPipe FaceMesh** | Vídeo | 468 landmarks faciais para análise de expressão |
| **Haar Cascade** (OpenCV) | Vídeo | Fallback quando MediaPipe falha |
| **Whisper base** (OpenAI) | Áudio | Transcrição local em português |
| **librosa** | Áudio | Extração de features vocais (pitch, energia, ZCR) |
| **Rule-based NLP** | Texto | Detecção de palavras-chave e padrões linguísticos |

## Requisitos

- Python 3.11+
- FFmpeg instalado no sistema (`brew install ffmpeg` no macOS)
- ~500MB de espaço para modelos (YOLOv8n + Whisper base)

## Instalação (macOS)

```bash
# 1. Clonar o repositório
git clone <repo-url>
cd Detec_facial

# 2. Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar FFmpeg (se não tiver)
brew install ffmpeg

# 4. Instalar dependências Python
pip install -r requirements.txt

# 5. Colocar o vídeo na pasta data/
mkdir -p data
# copie o vídeo .mp4 para data/
```

## Execução

```bash
# Análise completa (vídeo + áudio)
python main_analysis.py

# Apenas vídeo
python -c "
from video_analysis import VideoAnalyzer
v = VideoAnalyzer('data/seu_video.mp4')
v.analyze()
v.generate_report()
"
```

## Relatórios Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `relatorio_video.json/txt` | Detalhes da análise visual (expressões, hematomas, marcas) |
| `relatorio_audio.json/txt` | Transcrição, keywords, features vocais |
| `relatorio_integrado.json/txt` | Fusão multimodal com score final e recomendações |

## Estrutura do Projeto

```
Detec_facial/
├── main_analysis.py      # Orquestrador: fusão multimodal
├── video_analysis.py     # YOLOv8 + MediaPipe + detecção de anomalias
├── audio_analysis.py     # Whisper + librosa + análise linguística
├── requirements.txt      # Dependências Python
├── pyproject.toml        # Metadados do projeto
├── data/                 # Vídeos para análise (não versionados)
├── tech-challenger.md    # Enunciado do Tech Challenge
└── README.md             # Este arquivo
```

## Execução Local vs. Nuvem

**Tudo roda 100% local**, sem necessidade de serviços pagos:

| Componente | Alternativa cloud | Solução local usada |
|------------|-------------------|---------------------|
| Speech-to-Text | Azure/Google Speech API | **Whisper** (OpenAI, gratuito) |
| Detecção de objetos | Azure Computer Vision | **YOLOv8** (ultralytics, gratuito) |
| Análise de sentimento | Azure Text Analytics | **Rule-based NLP** (custom) |
| Face detection | Azure Face API | **MediaPipe** (Google, gratuito) |

## Linhas de Apoio

- **CVV** (saúde mental): 188 (24h, gratuito)
- **Central da Mulher** (violência): 180 (24h, gratuito)
- **SAMU**: 192
- **Disque Direitos Humanos**: 100

## O que foi alterado em relação à versão original

### Problemas encontrados na versão anterior

- Arquivo `simple_video_analysis.py` (476 linhas) era ~90% cópia do `video_analysis.py`
- 92 chamadas `print()` espalhadas nos 3 arquivos, sem controle de nível
- Transcrição de áudio dependia do **Google Speech API** (necessita internet, instável)
- Nenhum modelo de detecção de pessoa — ia direto para o rosto sem validar se havia alguém no frame
- Vocabulário de keywords limitado a depressão genérica, sem termos de saúde da mulher
- Sem análise de features vocais (pitch, energia, pausas)
- Sem fusão multimodal ponderada entre vídeo e áudio
- Sem sistema de classificação de risco por níveis
- Dependências desnecessárias (`matplotlib`, `imageio-ffmpeg`, `SpeechRecognition`)
- Indentação inconsistente em `video_analysis.py` (erros de compilação)
- Relatórios gerados com nomes genéricos (`analysis_report.json`)

### O que foi adicionado/corrigido

| Mudança | Antes | Depois |
|---------|-------|--------|
| Detecção de pessoa | Nenhuma | **YOLOv8n** como pré-filtro em cada frame |
| Transcrição de voz | Google Speech API (nuvem) | **Whisper base** (local, offline, gratuito) |
| Análise vocal | Não existia | **librosa** — pitch, energia, ZCR, detecção de pausas |
| Saída no terminal | 92 `print()` | 0 prints, **24 chamadas `logging`** com nível e timestamp |
| Arquivo duplicado | `simple_video_analysis.py` (476 linhas) | Deletado — fallback Haar integrado no `video_analysis.py` |
| Fusão dos resultados | Soma direta de scores | Score ponderado: **visual × 0.4 + áudio × 0.6** |
| Classificação de risco | Não existia | 4 níveis: BAIXO / MODERADO / ALTO / MUITO ALTO |
| Vocabulário | ~50 termos genéricos | +20 termos: pós-parto, violência doméstica, ansiedade gestacional, fadiga hormonal |
| Hesitações | Não detectava | Detecta: "né", "tipo", "assim", "ahn" |
| Detecção de hematomas | Ranges HSV genéricos (muitos falsos positivos) | Ranges **calibrados** + filtros morfológicos + área mínima |
| Relatórios | `analysis_report.json` | `relatorio_video`, `relatorio_audio`, `relatorio_integrado` (JSON + TXT) |
| Recursos de apoio | Não incluía | CVV 188, Central da Mulher 180, SAMU 192 |
| Total de linhas | 1.749 (4 arquivos) | **759** (3 arquivos) — redução de 57% |
