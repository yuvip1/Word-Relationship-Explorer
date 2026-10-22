# Word Relationship Explorer

A modern NLP mini project built with Streamlit and NLTK WordNet.

## Features
- Search any English word
- Find synonyms, antonyms, hypernyms, and hyponyms
- Interactive semantic relationship graph
- Word sense analysis with definitions and examples
- Search history and favorites
- Export results as CSV and JSON

## Project Structure
```text
word_relationship_explorer/
├── app.py
├── nlp_engine.py
├── requirements.txt
└── README.md
```

## How to Run in VS Code
1. Open this folder in VS Code.
2. Open terminal.
3. Create virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate it:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the project:
   ```bash
   streamlit run app.py
   ```

## Notes
- The first run may download NLTK WordNet data automatically.
- Use simple English words for best results.

## NLP Concepts Used
- Lexical semantics
- WordNet ontology
- Semantic relationship extraction
- Basic input normalization
- Interactive visualization
