"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This module provides utility functions for text cleaning and decoding. These functions are used to handle
Unicode characters, unwanted symbols, and other text normalization tasks.

Functions:
1. clean_text: Cleans and decodes text to handle Unicode characters and unwanted symbols.

Usage:
    from utilities.text_utils import clean_text
"""

import re
import html

def clean_text(text: str) -> str:
    """
    Cleans and decodes text to handle Unicode characters and unwanted symbols.

    Args:
        text (str): The text to be cleaned and decoded.

    Returns:
        str: The cleaned and decoded text.
    """
    text = text.encode('utf-8').decode('unicode_escape')  # Decode Unicode escape sequences
    text = html.unescape(text)                            # Unescape HTML entities
    text = re.sub(r"\u2019", "'", text)                   # Replace right single quotation mark with apostrophe
    text = re.sub(r"\u2014", "-", text)                   # Replace em dash with hyphen
    text = re.sub(r"\u201c", '"', text)                   # Replace left double quotation mark with double quote
    text = re.sub(r"\u201d", '"', text)                   # Replace right double quotation mark with double quote
    text = re.sub(r"\u2026", "...", text)                 # Replace ellipsis with three dots
    text = re.sub(r"[^a-zA-Z0-9\s.,?!'\":-]", "", text)   # Remove unwanted characters except ':' and '-
    text = re.sub(r"\s+", " ", text).strip()              # Remove extra whitespace
    return text
