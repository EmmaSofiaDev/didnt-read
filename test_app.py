"""
test_app.py - Verification and regression test suite for Didn't Read.
Tests text extraction, sentence segmentation, multi-label parsing, and FastAPI endpoints.
"""

import os
import json
from fastapi.testclient import TestClient
from app import app, split_into_sentences, parse_labels, LABEL_NAMES, get_checkpoint_model_path

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "didnt-read"

def test_home_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "Paste a Terms of Service link" in res.text
    assert "Best Use of Tinker" in res.text

def test_eval_page():
    res = client.get("/eval")
    assert res.status_code == 200
    assert "Tinker Fine-Tuning vs. Zero-Shot Baseline" in res.text
    assert "Limitation of liability" in res.text

def test_api_results():
    res = client.get("/api/results")
    assert res.status_code == 200
    data = res.json()
    assert "baseline" in data
    assert "finetune" in data
    assert data["finetune"]["macro_f1"] > 0.8

def test_sentence_segmentation():
    sample_legal_text = (
        "Welcome to our service. You agree not to reverse engineer the code. "
        "Any dispute arising under these terms shall be settled through binding arbitration in San Francisco, CA. "
        "We reserve the right to terminate accounts at our sole discretion without notice."
    )
    sentences = split_into_sentences(sample_legal_text)
    assert len(sentences) >= 3
    assert any("arbitration" in s.lower() for s in sentences)
    assert any("terminate" in s.lower() for s in sentences)

def test_label_parsing():
    # Test Fair
    assert parse_labels("Fair") == []
    assert parse_labels("This clause is completely Fair.") == []
    
    # Test single label
    assert "Arbitration" in parse_labels("Arbitration")
    assert "Limitation of liability" in parse_labels("Limitation of liability")
    
    # Test multi-label
    both = parse_labels("Choice of law, Jurisdiction")
    assert "Choice of law" in both
    assert "Jurisdiction" in both

def test_checkpoint_loaded():
    chk_path = get_checkpoint_model_path()
    assert chk_path is not None
    assert chk_path.startswith("tinker://")
    assert "didnt-read-unfair-tos-lora" in chk_path

if __name__ == "__main__":
    test_health_endpoint()
    test_home_page()
    test_eval_page()
    test_api_results()
    test_sentence_segmentation()
    test_label_parsing()
    test_checkpoint_loaded()
    print("[+] All 7/7 verification tests PASSED!")
