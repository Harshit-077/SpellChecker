"""
Compare All Three Spell Checker APIs
=====================================
Tests and compares results from:
1. Edit-Distance (port 8002)
2. Neural (port 8001)
3. Hybrid (port 8000)
"""

import requests
import json
from typing import Dict

SERVERS = {
    "hybrid": "http://localhost:8000",
    "neural": "http://localhost:8001",
    "edit-distance": "http://localhost:8002"
}

def test_server(name: str, url: str, text: str) -> Dict:
    """Test a single server"""
    try:
        response = requests.post(
            f"{url}/api/spell-check",
            json={"text": text},
            timeout=5
        )
        return {
            "status": "✓",
            "result": response.json()
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "✗",
            "error": "Server not running"
        }
    except Exception as e:
        return {
            "status": "✗",
            "error": str(e)
        }

def compare_all_servers(test_text: str):
    """Compare all three servers on the same text"""
    print("=" * 80)
    print(f"INPUT: {test_text}")
    print("=" * 80)
    
    results = {}
    
    for name, url in SERVERS.items():
        print(f"\n[{name.upper()}] Testing...")
        result = test_server(name, url, test_text)
        results[name] = result
        
        if result["status"] == "✓":
            data = result["result"]
            print(f"  Status: {result['status']}")
            print(f"  Corrected: {data['corrected']}")
            print(f"  Changed: {data['changed']}")
        else:
            print(f"  Status: {result['status']}")
            print(f"  Error: {result['error']}")
    
    return results

def detailed_comparison(test_words: list):
    """Show detailed word-by-word comparison"""
    print("\n" + "=" * 80)
    print("DETAILED WORD-BY-WORD COMPARISON")
    print("=" * 80)
    
    for word in test_words:
        print(f"\n{'-'*80}")
        print(f"Word: {word}")
        print(f"{'-'*80}")
        
        results = {}
        for name, url in SERVERS.items():
            result = test_server(name, url, word)
            if result["status"] == "✓":
                corrected = result["result"]["corrected"]
                results[name] = corrected
            else:
                results[name] = f"[{result['error']}]"
        
        # Display comparison
        print(f"  Edit-Distance:  {word:15s} → {results.get('edit-distance', 'N/A'):15s}")
        print(f"  Neural:         {word:15s} → {results.get('neural', 'N/A'):15s}")
        print(f"  Hybrid:         {word:15s} → {results.get('hybrid', 'N/A'):15s}")
        
        # Check agreement
        corrected_words = [v for v in results.values() if not v.startswith('[')]
        if len(set(corrected_words)) == 1:
            print(f"  → All agree: {corrected_words[0]} ✓")
        else:
            print(f"  → Disagreement!")

def check_server_health():
    """Check health of all servers"""
    print("=" * 80)
    print("SERVER HEALTH CHECK")
    print("=" * 80)
    
    for name, url in SERVERS.items():
        try:
            response = requests.get(f"{url}/", timeout=2)
            info = response.json()
            print(f"\n✓ {name.upper()} (Port {url.split(':')[-1]})")
            print(f"  Status: Running")
            if 'method' in info:
                print(f"  Method: {info['method']}")
            if 'accuracy' in info:
                print(f"  Accuracy: {info['accuracy']}")
        except requests.exceptions.ConnectionError:
            print(f"\n✗ {name.upper()} (Port {url.split(':')[-1]})")
            print(f"  Status: Not running")
            print(f"  Action: Start with 'python server_{name.replace('-', '')}.py'")
        except Exception as e:
            print(f"\n? {name.upper()}")
            print(f"  Error: {e}")

# ============================================================================
# MAIN TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HINDI SPELL CHECKER - THREE-WAY COMPARISON")
    print("=" * 80)
    
    # Check server health first
    check_server_health()
    
    # Test 1: Simple words
    print("\n\n" + "=" * 80)
    print("TEST 1: SIMPLE MISSPELLINGS")
    print("=" * 80)
    
    simple_words = ["भारतय", "सरकारि", "पानि"]
    detailed_comparison(simple_words)
    
    # Test 2: Complex corrections
    print("\n\n" + "=" * 80)
    print("TEST 2: COMPLEX CORRECTIONS")
    print("=" * 80)
    
    complex_words = ["विदयालय", "समसया", "परवरतन"]
    detailed_comparison(complex_words)
    
    # Test 3: Full sentence
    print("\n\n" + "=" * 80)
    print("TEST 3: FULL SENTENCE")
    print("=" * 80)
    
    test_sentence = "भारतय सरकारि विदयालय में पानि की समसया है"
    compare_all_servers(test_sentence)
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Server Comparison:

1. EDIT-DISTANCE (Port 8002)
   - Method: Levenshtein algorithm
   - Dictionary: 90k words
   - Speed: Very Fast ⚡⚡⚡
   - Accuracy: ~55%
   - Best for: Simple typos, dictionary words

2. NEURAL (Port 8001)
   - Method: Seq2Seq + Attention
   - Parameters: 8.3M
   - Speed: Moderate ⚡⚡
   - Accuracy: ~68%
   - Best for: Complex errors, morphology

3. HYBRID (Port 8000)
   - Method: Edit-Distance + Neural fusion
   - Speed: Moderate ⚡⚡
   - Accuracy: ~75-80%
   - Best for: Overall best results

Recommendation: Use HYBRID for production!
""")
    print("=" * 80)
