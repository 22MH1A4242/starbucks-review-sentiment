import re

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return " ".join(tokens)
