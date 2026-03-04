# 🎯 Sistema Multimodal de Triagem — Saúde da Mulher

Sistema inteligente de análise multimodal (vídeo + áudio) para **detecção precoce** de sinais de depressão, violência doméstica e problemas de saúde da mulher.

> ⚠️ **Ferramenta de triagem automatizada** — NÃO substitui avaliação profissional por especialista.

## 🎓 Objetivos (POS TECH — Tech Challenge Fase 4)

| # | Objetivo | Implementação |
|---|----------|---|
| 2 | Identificar sinais de violência doméstica | Detecção de hematomas/marcas via YOLOv8 + análise de cor HSV |
| 3 | Monitorar bem-estar psicológico feminino | Análise de expressões faciais (MediaPipe) + processamento de linguagem (Whisper) |
| 5 | Detecção de anomalias para monitoramento preventivo | Pipeline automatizado com sistema de alertas estratificado por nível de risco |

## 🔄 Arquitetura — Fluxo Multimodal

```
Arquivo de vídeo (MP4)
  │
  ├─► YOLOv8n (detecção de pessoa no frame)
  │     └─► MediaPipe FaceMesh (468 landmarks faciais)
  │           ├─► Análise de expressão (eye aspect ratio, mouth ratio)
  │           └─► HSV (detecção de hematomas, marcas de varizes)
  │
  ├─► FFmpeg (extração de áudio)
  │     └─► Whisper base (transcrição local em português)
  │           ├─► Análise linguística (keywords: depressão, violência, pós-parto)
  │           └─► Padrões vocais (hesitações, ritmo de fala)
  │
  └─► librosa (análise de features vocais)
        ├─► Pitch (frequência fundamental)
        ├─► Energia (intensidade da voz)
        └─► Pausas e silêncios

  ╔═══════════════════════════════════════════════════════════╗
  ║  FUSÃO MULTIMODAL                                         ║
  ║  Score Final = score_visual × 0.4 + score_audio × 0.6   ║
  ║  Classificação: BAIXO / MODERADO / ALTO / MUITO ALTO     ║
  ╚═══════════════════════════════════════════════════════════╝
  
  └─► Relatório integrado com nível de risco e recomendações
```

## 🧠 Modelos e Tecnologias

| Modelo | Tipo | Função |
|--------|------|--------|
| **YOLOv8n** (ultralytics) | Visão | Detecção de pessoa no frame (pré-filtro) |
| **MediaPipe FaceMesh** | Visão | 468 landmarks faciais para análise de expressão |
| **Haar Cascade** (OpenCV) | Visão | Fallback automático quando MediaPipe falha |
| **Whisper base** (OpenAI) | Áudio | Transcrição local em português (offline) |
| **librosa** | Áudio | Extração de features vocais (pitch, energia, zero crossing rate) |
| **Rule-based NLP** | Processamento | Detecção de keywords e padrões linguísticos |

## 📋 Requisitos do Sistema

- **Python:** 3.11 ou superior
- **FFmpeg:** Necessário para processing de áudio (instale via gerenciador de pacotes)
- **Espaço em disco:** ~500 MB para modelos (YOLOv8n + Whisper base)
- **Memória RAM:** Mínimo 4 GB recomendado

### Instalação de dependências do sistema

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg python3-dev
```

**macOS (com Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Baixe FFmpeg em https://ffmpeg.org/download.html ou use `choco install ffmpeg`

## 🚀 Instalação e Setup

### 1. Clonar o repositório
```bash
git clone <repo-url>
cd Detec_facial
```

### 2. Criar ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou, no Windows:
# venv\Scripts\activate
```

### 3. Instalar dependências Python
```bash
pip install --upgrade pip
pip install -r requirements.txt
# ou via pyproject.toml:
# pip install -e .
```

### 4. Preparar dados de entrada
```bash
mkdir -p data
# Coloque seu vídeo .mp4 na pasta data/
# Exemplo: data/video_exemplo.mp4
```

### 5. Baixar modelos (primeira execução)
Os modelos são baixados automaticamente na primeira execução:
- YOLOv8n: ~35 MB
- Whisper base: ~140 MB

## ▶️ Como Usar

### Análise Completa (Vídeo + Áudio)
```bash
python src/main_analysis.py
```
Processa o vídeo em `data/` e gera 3 relatórios:
- `relatorio_video.json/txt` — Análise visual
- `relatorio_audio.json/txt` — Análise de voz
- `relatorio_integrado.json/txt` — Resultado final com risco e recomendações

### Apenas Análise de Vídeo
```bash
# No Python intereativo ou script:
from src.video_analysis import VideoAnalyzer
analyzer = VideoAnalyzer('data/seu_video.mp4')
report = analyzer.analyze()
analyzer.generate_report()
```

### Apenas Análise de Áudio
```bash
from src.audio_analysis import AudioAnalyzer
analyzer = AudioAnalyzer('data/seu_video.mp4')
report = analyzer.analyze()
analyzer.generate_report()
```

### Parâmetros customizáveis
Se quiser ajustar pesos e limiares, edite as constantes em `src/main_analysis.py`:

```python
WEIGHT_VIDEO = 0.4  # Peso da análise visual
WEIGHT_AUDIO = 0.6  # Peso da análise de áudio
```

## 📊 Relatórios Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `relatorio_video.json` | Análise frame-by-frame: expressões, hematomas, variações de cor |
| `relatorio_video.txt` | Versão legível com indicadores e resumo visual |
| `relatorio_audio.json` | Transcrição completa, keywords, features vocais (pitch, energia) |
| `relatorio_audio.txt` | Versão legível da análise de voz |
| `relatorio_integrado.json` | **Score final**, nível de risco, recomendações acionáveis |
| `relatorio_integrado.txt` | Resumo formatado com orientações para profissional |

### Exemplo de saída (relatorio_integrado.txt):
```
══════════════════════════════════════════════════════════
             RELATÓRIO INTEGRADO — RESULTADO
══════════════════════════════════════════════════════════

Score Visual: 6.2
Score Áudio: 8.1
Score Final: 7.4

CLASSIFICAÇÃO: MODERADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMENDAÇÕES:
✓ ATENÇÃO: Alguns indicadores presentes
✓ Converse com pessoas de confiança
✓ Considere apoio psicológico
✓ CVV: 188 (24h, gratuito)
```

## 📁 Estrutura do Projeto

```
Detec_facial/
├── src/
│   ├── main_analysis.py          # Orquestrador: fusão de scores visuais + áudio
│   ├── video_analysis.py         # YOLOv8 + MediaPipe + detecção de anomalias (HSV)
│   └── audio_analysis.py         # Whisper + librosa + análise linguística
├── data/                         # Vídeos para análise (não versionados no git)
├── __pycache__/                  # Cache Python (gerado automaticamente)
├── main_analysis.py              # Legacy — utilizar src/main_analysis.py
├── pyproject.toml                # Metadados do projeto e dependências
├── README.md                     # Este arquivo
├── requirements.txt              # Dependências Python (compatível com pip)
├── tech-challenger.md            # Enunciado original do Tech Challenge
├── yolov8n.pt                    # Modelo YOLOv8n (baixado automaticamente)
├── relatorio_*.json/txt          # Saídas geradas pela análise
└── .gitignore                    # Arquivos ignorados pelo git
```

### Notas sobre a estrutura:
- **src/** contém o código principal e está pronto em produção
- **data/** é onde você coloca vídeos de entrada (não é versionado)
- **requirements.txt** é gerado a partir de pyproject.toml (use um ou outro)
- Relatórios são salvos na raiz do projeto para fácil acesso

## 🌐 Execução Local vs. Nuvem

**Tudo roda 100% local**, sem necessidade de serviços pagos:

| Componente | Alternativa cloud | Solução local usada |
|------------|-------------------|---------------------|
| Speech-to-Text | Azure/Google Speech API | **Whisper** (OpenAI, gratuito) |
| Detecção de objetos | Azure Computer Vision | **YOLOv8** (ultralytics, gratuito) |
| Análise de sentimento | Azure Text Analytics | **Rule-based NLP** (custom) |
| Face detection | Azure Face API | **MediaPipe** (Google, gratuito) |

## 🆘 Recursos de Apoio e Suporte

Se você ou alguém que conhece está em situação de risco, procure ajuda:

- 🧠 **CVV** (Valorização da Vida) — Prevenção do suicídio e saúde mental
  - Telefone: **188** | 24h | Chamada gratuita
  
- 👩 **Central de Atendimento à Mulher** — Violência doméstica e direitos
  - Telefone: **180** | 24h | Ligação gratuita
  
- 🚑 **SAMU** — Emergências médicas
  - Telefone: **192** | 24h | Serviço de ambulância
  
- ⚖️ **Disque Direitos Humanos** — Denúncias de violação de direitos
  - Telefone: **100** | 24h | Ligação gratuita

> **⚠️ AVISO IMPORTANTE:** Este sistema é uma **ferramenta de triagem** e não substitui avaliação profissional. Se você está em situação de emergência, ligue imediatamente para o 192 (SAMU) ou 180 (Central da Mulher).

## 🔧 Troubleshooting

### Erro: "Cannot import name 'VideoAnalyzer'"
```bash
# Certifique-se de estar rodando no diretório raiz:
cd /home/turt/Fiap/Tech_challenge_4/Detec_facial
python -m src.main_analysis  # ou python src/main_analysis.py
```

### Erro: "FFmpeg not found"
```bash
# Instale FFmpeg no seu sistema operacional (veja seção Requisitos)
which ffmpeg  # Verifique se está instalado
```

### Memória insuficiente
Se receber erro de OOM ao processar vídeos:
- Reduza a qualidade do vídeo de entrada
- Processe um vídeo menor para teste
- Aumente a RAM disponível do sistema

### Modelos não baixaram
Os modelos são baixados automaticamente na primeira execução. Caso falhe:
```bash
yolo detect train model=yolov8n.pt  # Força download do YOLOv8
python -c "import whisper; whisper.load_model('base')"  # Força download de Whisper
```

## Histórico de Mudanças

### O que foi alterado em relação à versão original

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
