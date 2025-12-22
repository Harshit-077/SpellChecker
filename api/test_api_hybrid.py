"""
Test Hybrid API Server
=======================
Demonstrates API usage for frontend integration
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 80)
print("TESTING HYBRID SPELL CHECKER API")
print("=" * 80)

# Test 1: Health Check
print("\n[1] Health Check")
print("-" * 80)
response = requests.get(f"{API_URL}/health")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# Test 2: API Info
print("\n[2] API Information")
print("-" * 80)
response = requests.get(f"{API_URL}/")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# Test 3: Spell Check Single Word
print("\n[3] Correct Single Word")
print("-" * 80)
response = requests.post(
    f"{API_URL}/api/correct-word",
    json={"text": "भारतय", "strategy": "hybrid"}
)
result = response.json()
print(f"Input: {result['input']}")
print(f"Corrected: {result['corrected']}")
print(f"Method: {result['details']['method_used']}")
print(f"Confidence: {result['details']['confidence']:.1%}")

# Test 4: Spell Check Full Text
print("\n[4] Spell Check Full Text")
print("-" * 80)
test_text = "भारतय सरकारि विदयालय में पानि की समसया है"
response = requests.post(
    f"{API_URL}/api/spell-check",
    json={"text": test_text, "strategy": "hybrid"}
)
result = response.json()

print(f"Original: {result['input']}")
print(f"Corrected: {result['corrected']}")
print(f"Changed: {result['changed']}")
print(f"\nWord-by-word details:")
for word in result['words']:
    if word['changed']:
        print(f"  {word['original']:15s} → {word['corrected']:15s} "
              f"[{word['method_used']}, {word['confidence']:.0%}]")

# Test 5: Compare Methods
print("\n[5] Compare All Methods")
print("-" * 80)
test_words = ["भारतय", "सरकारि", "पानि"]
for word in test_words:
    response = requests.post(
        f"{API_URL}/api/compare-methods",
        json={"text": word}
    )
    result = response.json()
    print(f"\n{word}:")
    print(f"  Edit-Distance: {result['results']['edit_distance']}")
    print(f"  Neural:        {result['results']['neural']}")
    print(f"  Hybrid:        {result['results']['hybrid']}")

# Test 6: Different Strategies
print("\n[6] Testing Different Strategies")
print("-" * 80)
word = "भारतय"
strategies = ["edit-distance", "neural", "hybrid"]

for strategy in strategies:
    response = requests.post(
        f"{API_URL}/api/correct-word",
        json={"text": word, "strategy": strategy}
    )
    result = response.json()
    print(f"{strategy:15s}: {word} → {result['corrected']}")

print("\n" + "=" * 80)
print("API TESTS COMPLETE")
print("=" * 80)
print("\nFrontend Integration Example:")
print("""
// JavaScript/React example
const correctText = async (text) => {
  const response = await fetch('http://localhost:8000/api/spell-check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, strategy: 'hybrid' })
  });
  
  const result = await response.json();
  console.log('Corrected:', result.corrected);
  console.log('Words:', result.words);
  
  return result;
};
""")
print("=" * 80)
