"""
Tests for search.py — CLI semantic search helper function.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from search import search_food


def test_search_food_empty_query():
    results = search_food("   ")
    assert results == [], "Empty query text should return empty list."


def test_search_food_structure():
    # Verify search returns structured dictionary fields when query is passed
    results = search_food("protein salad", top_k=1)
    if results:
        item = results[0]
        assert "name" in item
        assert "cuisine" in item
        assert "calories" in item
        assert "dietary_type" in item
