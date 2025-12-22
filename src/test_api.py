import requests
import json

def test_spell_check(text):
    url = "http://localhost:8000/api/spell-check"
    response = requests.post(url, json={"text": text})
    result = response.json()
    
    print(f"Input:     {result['input']}")
    print(f"Corrected: {result['corrected']}")
    print(f"Changed:   {result['changed']}")
    print()

# Test cases
test_spell_check("भारतय")
test_spell_check("भारत विद्यालय")
test_spell_check("परिवर्तन")