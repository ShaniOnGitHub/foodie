"""
Tests for ingest.py — record_to_text formatting and the ingestion count gate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import record_to_text, EXPECTED_COUNT, DATA_PATH


def test_record_to_text_includes_all_fields():
    item = {
        "name": "Miso Soup",
        "dietary_type": "vegan",
        "cuisine": "Japanese",
        "calories": 45,
        "cooking_method": "steamed",
        "taste_profile": "umami, light",
        "description": "A warm broth based starter.",
    }
    text = record_to_text(item)
    assert "Miso Soup" in text
    assert "vegan" in text
    assert "Japanese" in text
    assert "45 calories" in text
    assert "steamed" in text
    assert "umami, light" in text
    assert "A warm broth based starter." in text


def test_expected_count_matches_dataset_size():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        foods = json.load(f)
    assert len(foods) == EXPECTED_COUNT
