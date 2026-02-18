# Sistema de Análise de Vídeos
## Detecção de Sinais de Depressão, Violência Doméstica e Problemas de Saúde

Este sistema utiliza inteligência artificial para analisar vídeos e detectar:
- **Sinais de Depressão**: Através de expressões faciais e análise de fala
- **Hematomas**: Possíveis indicadores de violência doméstica
- **Marcas e Machucados**: Sinais de problemas de saúde

## ⚠️ AVISO IMPORTANTE

Esta ferramenta é um **sistema de apoio e triagem**, **NÃO substitui avaliação profissional**. 
Em caso de risco, procure ajuda profissional imediatamente:
- **CVV (Valorização da Vida)**: 188 (24h)
- **Central de Atendimento à Mulher**: 180
- **SAMU**: 192
- **Polícia Militar**: 190

## 📋 Requisitos

- Python 3.8 ou superior
- FFmpeg (para extração de áudio)

## 🚀 Instalação

### 1. Instalar Python
Certifique-se de ter Python 3.8+ instalado: `python --version`

### 2. Instalar Dependências Python
```bash
pip install -r requirements.txt
```

### 3. Instalar FFmpeg (Windows)

**Opção 1 - Usando Chocolatey:**
```bash
choco install ffmpeg
```

**Opção 2 - Download Manual:**
1. Baixe o FFmpeg em: https://ffmpeg.org/download.html
2. Extraia para `C:\ffmpeg`
3. Adicione `C:\ffmpeg\bin` ao PATH do sistema

Para verificar a instalação:
```bash
ffmpeg -version
```

## 📁 Estrutura do Projeto

```
Detec_facial/
├── data/                           # Pasta com vídeos para análise
│   └── YTDown.com_YouTube_Media_5t_FoFzVcsA_001_720p.mp4
├── main_analysis.py                # Script principal (análise integrada)
├── video_analysis.py               # Análise visual (expressões, hematomas)
├── audio_analysis.py               # Análise de áudio/fala
├── requirements.txt                # Dependências Python
└── README.md                       # Este arquivo
```

## 💻 Como Usar

### Análise Completa (Recomendado)

Execute o script principal que realiza análise integrada de vídeo e áudio:

```bash
python main_analysis.py
```

Este comando vai:
1. Analisar expressões faciais frame a frame
2. Detectar possíveis hematomas e marcas
3. Extrair e transcrever o áudio
4. Analisar a fala para indicadores de depressão
5. Gerar relatórios consolidados

### Análise Apenas de Vídeo

Se quiser analisar apenas aspectos visuais:

```bash
python video_analysis.py
```

### Análise Apenas de Áudio

Se quiser analisar apenas a fala:

```bash
python audio_analysis.py
```

## 📊 Relatórios Gerados

Após a execução, serão criados os seguintes arquivos:

### Relatórios Principais:
- **RELATORIO_FINAL_INTEGRADO.json** - Relatório completo em JSON
- **RELATORIO_FINAL_INTEGRADO.txt** - Relatório completo legível

### Relatórios Detalhados:
- **analysis_report.json** / **analysis_report.txt** - Detalhes da análise visual
- **audio_analysis_report.json** / **audio_analysis_report.txt** - Detalhes da análise de áudio

## 🔍 O Que o Sistema Analisa

### 1. Análise de Depressão

**Indicadores Visuais:**
- Abertura dos olhos (cansaço, falta de energia)
- Expressão da boca (falta de sorriso, tristeza)
- Posição das sobrancelhas
- Expressões faciais em geral

**Indicadores na Fala:**
- Palavras-chave relacionadas à depressão (tristeza, solidão, desespero, etc.)
- Padrões linguísticos negativos
- Tom de voz (pitch baixo)
- Energia vocal
- Uso excessivo de primeira pessoa (ruminação)

### 2. Detecção de Hematomas (Violência Doméstica)

- Identifica áreas com coloração:
  - Roxa/azulada (hematomas frescos)
  - Amarelada/esverdeada (hematomas antigos)
  - Escura (hematomas recentes)
- Mapeia localização dos hematomas no rosto
- Calcula score de risco

### 3. Detecção de Marcas e Machucados

- Identifica marcas vermelhas
- Detecta possíveis ferimentos
- Identifica irritações cutâneas
- Sugere avaliação médica quando necessário

## 📈 Interpretação dos Scores

### Score de Depressão:
- **< 3**: Baixo risco
- **3-8**: Risco moderado - Atenção recomendada
- **8-15**: Alto risco - Avaliação profissional recomendada
- **> 15**: Muito alto risco - Ação imediata necessária

### Score de Hematomas:
- **< 5**: Baixo risco
- **5-15**: Risco moderado - Investigação recomendada
- **> 15**: Alto risco - Avaliação urgente recomendada

## ⚙️ Configurações Avançadas

### Ajustar Taxa de Amostragem

No código, você pode ajustar `sample_rate` em `video_analysis.py`:

```python
results = analyzer.analyze_video(sample_rate=30)  # Processa 1 frame a cada 30
```

- **sample_rate=10**: Análise mais detalhada (mais lenta)
- **sample_rate=30**: Análise equilibrada (padrão)
- **sample_rate=60**: Análise mais rápida (menos detalhada)

### Processar Outros Vídeos

Modifique o caminho do vídeo nos scripts:

```python
video_path = 'data/seu_video.mp4'
```

## 🛠️ Solução de Problemas

### Erro: FFmpeg não encontrado
```
AVISO: ffmpeg não encontrado
```
**Solução**: Instale o FFmpeg seguindo as instruções acima.

### Erro: SpeechRecognition não instalado
```
AVISO: SpeechRecognition não instalado
```
**Solução**: 
```bash
pip install SpeechRecognition
```

### Erro: librosa não instalado
```
AVISO: librosa não instalado
```
**Solução**: 
```bash
pip install librosa
```

### Análise de áudio não funciona
- Verifique se o FFmpeg está instalado corretamente
- Teste: `ffmpeg -version`
- Certifique-se de que o vídeo tem áudio

### Vídeo não é processado
- Verifique se o arquivo de vídeo existe na pasta `data/`
- Verifique o formato (MP4, AVI, MOV são suportados)
- Verifique se o caminho está correto

## 📚 Tecnologias Utilizadas

- **OpenCV**: Processamento de vídeo e imagem
- **MediaPipe**: Detecção facial e landmarks
- **SpeechRecognition**: Transcrição de áudio
- **Librosa**: Análise de características vocais
- **NumPy**: Operações numéricas
- **FFmpeg**: Extração e processamento de áudio

## 🔐 Privacidade e Ética

- Todos os dados são processados **localmente** na sua máquina
- Nenhuma informação é enviada para servidores externos (exceto transcrição de áudio via Google Speech API)
- Use esta ferramenta de forma ética e responsável
- Respeite a privacidade das pessoas nos vídeos
- Obtenha consentimento antes de analisar vídeos de terceiros

## 🤝 Suporte e Recursos

### Linhas de Apoio (Brasil):

**Saúde Mental:**
- CVV - Centro de Valorização da Vida: **188** (24h)
- CAPS - Centro de Atenção Psicossocial (busque o mais próximo)
- SAMU: **192**

**Violência Doméstica:**
- Central de Atendimento à Mulher: **180** (24h)
- Polícia Militar: **190**
- Delegacia da Mulher (busque a mais próxima)
- Disque Direitos Humanos: **100**

**Saúde Geral:**
- SAMU: **192**
- UBS - Unidade Básica de Saúde (busque a mais próxima)

## 📝 Limitações

- A detecção é baseada em padrões visuais e auditivos, não é 100% precisa
- Fatores como iluminação, qualidade do vídeo e ângulo da câmera afetam os resultados
- Não substitui avaliação profissional médica ou psicológica
- Deve ser usada como ferramenta de **triagem e apoio**, não diagnóstico

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.

## ⚠️ Disclaimer

Esta ferramenta NÃO substitui profissionais de saúde, psicólogos, assistentes sociais ou autoridades competentes. Em situações de risco, procure ajuda profissional imediatamente.

---

**Desenvolvido como ferramenta de apoio para detecção precoce de situações de risco.**
