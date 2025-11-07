import re
from typing import List, Tuple

# regex que captura palabras (incluye apóstrofos) y tokens de puntuación como unidades separadas
_WORD_PUNCT_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+|[¿¡]|[^\w\s]", flags=re.UNICODE)

OPEN_PUNCT = {"¿", "¡"}
CLOSE_PUNCT = {",", ".", "?", "!"}  # podemos extender si hace falta

def split_words_and_punct(text):
    """
    Separa texto en una lista donde las unidades pueden ser:
      - palabra (incluye apostrofes, dígitos)
      - signos de puntuación como tokens separados (.,,¿,?,!, etc)
    Ej: "¿Cómo estás?" -> ["¿", "Cómo", "estás", "?"]
    """
    return _WORD_PUNCT_RE.findall(text)

def is_punct_token(tok: str) -> bool:
    """Verdadero si tok es un token de puntuación (no palabra)"""
    return bool(re.match(r"^[^\w\s]$|^[¿¡]$", tok, flags=re.UNICODE))

def cap_class_for_word(word: str) -> int:
    """
    Devuelve clase de capitalización:
      0: todo minúsculas
      1: primera letra mayúscula (incluye palabras de 1 letra)
      2: algunas (pero no todas) mayúsculas (mixed case, e.g. McDonald's)
      3: todas mayúsculas (más de una letra) (e.g., UBA, NASA)
    Se aplica sobre la palabra original (incluye apóstrofes).
    """
    letters = [c for c in word if c.isalpha()]
    if len(letters) == 0:
        return 0
    all_lower = all(c.islower() for c in letters)
    all_upper = all(c.isupper() for c in letters)
    if all_lower:
        return 0
    if all_upper and len(letters) > 1:
        return 3
    # first letter uppercase and rest lowercase?
    first_letter = next((c for c in word if c.isalpha()), None)
    if first_letter and first_letter.isupper() and all(c.islower() for c in letters[1:]):
        return 1
    # otherwise mixed
    return 2

def build_word_level_labels(words_and_punct: List[str]) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Dado words_and_punct (lista intercalando palabras y signos de puntuación) devuelve:
      - words_norm: lista de palabras *no punteadas* (sin tokens de puntuación), en su forma original
      - labels_punt_ini: lista binaria por palabra (0/1) indicando apertura de pregunta ¿ (u otro marcado de apertura)
      - labels_punt_fin: lista 0..3 por palabra (0=none,1=',' 2='.' 3='?')
      - labels_cap: lista 0..3 por palabra (capitalization class)
    Reglas:
      - '¿' o '¡' se marcan como punt_inicial=1 para la palabra siguiente no-puntuación.
      - ',', '.', '?' (y '!') se marcan como punt_final para la palabra precedente no-puntuación.
    """
    n = len(words_and_punct)
    # índices de palabras (no-puntuación)
    word_indices = [i for i, w in enumerate(words_and_punct) if not is_punct_token(w)]
    words_norm = [words_and_punct[i] for i in word_indices]
    m = len(words_norm)
    punct_ini = [0] * m
    punct_fin = [0] * m  # 0 none, 1 comma, 2 period, 3 question
    cap = [0] * m

    # maps from word_positions in words_norm back to original index
    orig_idx_of_wordpos = {pos: word_indices[pos] for pos in range(m)}

    # Build helper: for each original index, map to wordpos or None
    orig_to_wordpos = {}
    for pos, orig_i in orig_idx_of_wordpos.items():
        orig_to_wordpos[orig_i] = pos

    # iterate original tokens and apply punctuation rules
    for i, tok in enumerate(words_and_punct):
        if tok in OPEN_PUNCT:
            # mark next non-punct word if any
            j = i + 1
            while j < n and is_punct_token(words_and_punct[j]):
                j += 1
            if j < n and j in orig_to_wordpos:
                wpos = orig_to_wordpos[j]
                punct_ini[wpos] = 1
        elif tok in CLOSE_PUNCT:
            # mark previous non-punct word
            j = i - 1
            while j >= 0 and is_punct_token(words_and_punct[j]):
                j -= 1
            if j >= 0 and j in orig_to_wordpos:
                wpos = orig_to_wordpos[j]
                if tok == ",":
                    punct_fin[wpos] = 1
                elif tok == ".":
                    punct_fin[wpos] = 2
                elif tok == "?":
                    punct_fin[wpos] = 3
                elif tok == "!":
                    # map '!' to period class (or decide autre). Aquí lo mapeo a 2 (punto).
                    punct_fin[wpos] = 2

    # capitalization classes
    for p in range(m):
        cap[p] = cap_class_for_word(words_norm[p])

    return words_norm, punct_ini, punct_fin, cap