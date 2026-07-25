# RAG Kubernetes Retrieval Comparison

Vergleich von BM25 und Embedding-basierter Vektorsuche in einer RAG-Pipeline

## Überblick

Dieses Projekt implementiert und vergleicht zwei Retrieval-Methoden für Kubernetes-Dokumente:

- **BM25 (Lexikalische Suche)**: Traditionelle TF-IDF-basierte Volltextsuche
- **Embedding-basierte Vektorsuche**: Semantische Ähnlichkeit mit hochdimensionalen Vektoren

Beide Systeme werden in eine RAG-Pipeline (Retrieval-Augmented Generation) mit einem lokalen LLM (Mistral via Ollama) integriert und mittels etablierter IR-Metriken evaluiert.

## Quick Start

### 1. Umgebung einrichten

```bash
# Python Virtual Environment erstellen
python -m venv venv

# Aktivieren
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Ollama & Mistral installieren

```bash
# Ollama herunterladen von https://ollama.ai
# Nach Installation Mistral Model pullen:
ollama pull mistral:7b

# Ollama starten (in separatem Terminal):
ollama serve
```

### 3. Sentence Transformers Model herunterladen

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 4. Daten vorbereiten

```bash
# Kubernetes YAML Dateien in data/kubernetes/ platzieren
mkdir -p data/kubernetes data/queries data/processed

# Ground Truth & Queries definieren (siehe data/queries/README.md)
```

### 5. Evaluation starten

```bash
python scripts/run_evaluation.py
```

### 6. Fragen mit LLM Anbindung stellen

```bash
python scripts/ask_question.py
```

## 📊 Evaluationsmetriken

Das Projekt evaluiert beide Retriever mit folgenden Metriken:

- **Precision@k**: Anteil relevanter Dokumente in Top-k Ergebnissen
- **Recall@k**: Anteil gefundener relevanter Dokumente
- **F1@k**: Harmonisches Mittel aus Precision und Recall
- **nDCG@k (Normalized Discounted Cumulative Gain)**: Ranking-Qualität mit Positionsgewichtung
- **MRR (Mean Reciprocal Rank)**: Position des ersten relevanten Dokuments
- **AP (Average Precision)**: Durchschnittliche Precision über alle relevanten Positionen hinweg

## 🧪 Testing

```bash
# Unit Tests ausführen
pytest tests/ -v

# Mit Coverage Report:
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Konfiguration

Zentrale Konfiguration in `src/config.py` und `.env`

## 📈 Ergebnisse

Nach erfolgreicher Evaluation werden folgende Outputs generiert:

- `results/metrics_comparison.csv` - Vergleichende Metriken
- `results/evaluation_results.json` - Detailergebnisse

## 🔗 References

- [Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama](https://ollama.ai/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
