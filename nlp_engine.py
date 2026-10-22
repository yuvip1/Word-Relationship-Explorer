from __future__ import annotations

import html
import re
from typing import Dict, List

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

POS_MAP = {
    "noun": wn.NOUN,
    "verb": wn.VERB,
    "adjective": wn.ADJ,
    "adverb": wn.ADV,
}


def download_nltk_resources() -> None:
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def normalize_word(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().lower())


def _format_word(raw: str) -> str:
    return raw.replace("_", " ")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def get_word_relationships(word: str, max_items: int = 12, pos_filter: str = "All") -> Dict:
    synsets = wn.synsets(word) if pos_filter == "All" else wn.synsets(word, pos=POS_MAP.get(pos_filter))

    synonyms: List[str] = []
    antonyms: List[str] = []
    hypernyms: List[str] = []
    hyponyms: List[str] = []
    definitions: List[str] = []
    senses: List[Dict] = []

    for syn in synsets[:8]:
        definitions.append(syn.definition())
        sense_info = {
            "name": syn.name(),
            "pos": syn.pos(),
            "definition": syn.definition(),
            "examples": syn.examples(),
            "lemmas": [_format_word(l.name()) for l in syn.lemmas()[:10]],
        }
        senses.append(sense_info)

        for lemma in syn.lemmas():
            synonyms.append(_format_word(lemma.name()))
            for ant in lemma.antonyms():
                antonyms.append(_format_word(ant.name()))

        for h in syn.hypernyms():
            for lemma in h.lemmas():
                hypernyms.append(_format_word(lemma.name()))

        for h in syn.hyponyms()[:15]:
            for lemma in h.lemmas()[:3]:
                hyponyms.append(_format_word(lemma.name()))

    synonyms = [x for x in _dedupe_keep_order(synonyms) if x != word][:max_items]
    antonyms = [x for x in _dedupe_keep_order(antonyms) if x != word][:max_items]
    hypernyms = [x for x in _dedupe_keep_order(hypernyms) if x != word][:max_items]
    hyponyms = [x for x in _dedupe_keep_order(hyponyms) if x != word][:max_items]
    definitions = _dedupe_keep_order(definitions)[:8]

    return {
        "word": word,
        "found": len(synsets) > 0,
        "synonyms": synonyms,
        "antonyms": antonyms,
        "hypernyms": hypernyms,
        "hyponyms": hyponyms,
        "definitions": definitions,
        "senses": senses,
    }


def build_relation_dataframe(center_word: str, result: Dict) -> pd.DataFrame:
    rows = []
    for rel in ["synonyms", "antonyms", "hypernyms", "hyponyms"]:
        for item in result.get(rel, []):
            rows.append(
                {
                    "source_word": center_word,
                    "relationship": rel[:-1] if rel.endswith("s") else rel,
                    "target_word": item,
                }
            )
    return pd.DataFrame(rows)


def build_network_html(center_word: str, result: Dict) -> str:
    color_map = {
        "synonyms": "#3b82f6",
        "antonyms": "#ef4444",
        "hypernyms": "#10b981",
        "hyponyms": "#f59e0b",
    }

    nodes = [
        {
            "id": center_word,
            "label": center_word,
            "title": center_word,
            "color": "#8b5cf6",
            "size": 30,
        }
    ]
    edges = []

    for rel in ["synonyms", "antonyms", "hypernyms", "hyponyms"]:
        for item in result.get(rel, []):
            nodes.append(
                {
                    "id": f"{rel}:{item}",
                    "label": item,
                    "title": f"{rel[:-1].capitalize()}: {item}",
                    "color": color_map[rel],
                    "size": 18,
                }
            )
            edges.append(
                {
                    "from": center_word,
                    "to": f"{rel}:{item}",
                    "label": rel[:-1],
                    "color": {"color": color_map[rel]},
                    "font": {"color": "#d1d5db", "size": 12},
                    "width": 2,
                }
            )

    import json

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset='utf-8'>
      <script type='text/javascript' src='https://unpkg.com/vis-network/standalone/umd/vis-network.min.js'></script>
      <style>
        body {{ margin:0; background:#0b1220; font-family:Arial, sans-serif; }}
        #network {{ width:100%; height:100vh; border-radius:18px; }}
      </style>
    </head>
    <body>
      <div id='network'></div>
      <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes, edges }};
        const options = {{
          nodes: {{
            shape: 'dot',
            font: {{ color: '#f8fafc', size: 16, face: 'Arial' }},
            borderWidth: 1,
            shadow: true
          }},
          edges: {{
            smooth: {{ type: 'dynamic' }},
            arrows: {{ to: false }}
          }},
          interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
          physics: {{
            enabled: true,
            stabilization: false,
            barnesHut: {{ gravitationalConstant: -3000, springLength: 145, springConstant: 0.05 }}
          }}
        }};
        new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """

