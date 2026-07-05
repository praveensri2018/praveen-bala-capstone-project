import os
import re
import json
import requests
import joblib
import pandas as pd
import jsonschema


os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if line.strip().startswith('LLM_API_KEY='):
                os.environ['LLM_API_KEY'] = line.strip().split('=', 1)[1].strip("'\"")


def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    if bool(re.search(email_pattern, text) or re.search(phone_pattern, text)):
        return True
    return False


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
        
    api_key = os.environ.get('LLM_API_KEY', 'dummy_key')
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print("API Error:", response.status_code, response.text)
            return None
    except Exception as e:
        print("request failed", e)
        return None


def main():
    print("- Testing PII -")
    print("Test 1:", call_llm("be helpful", "my email is test@test.com"))
    
    print("\n- Testing LLM -")
    print("Test 2:", call_llm("be helpful", "Reply with only the word: hello"))


    print("\n- Loading Model -")
    model = joblib.load('../part3/best_model.pkl')
    
    
    schema = {
        "type": "object",
        "properties": {
            "prediction_label": {"type": ["string", "number"]},
            "confidence_level": {"type": "string"},
            "top_reason": {"type": "string"},
            "second_reason": {"type": "string"},
            "next_step": {"type": "string"}
        },
        "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
    }


    fallback = {
        "prediction_label": None,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "next_step": None
    }


    inputs = [
        {'pclass': 1, 'age': 40, 'sibsp': 0, 'parch': 0, 'class': 2, 'adult_male': 1, 'sex_male': 1, 'embarked_Q': 0, 'embarked_S': 1, 'embark_town_Queenstown': 0, 'embark_town_Southampton': 1, 'who_man': 1, 'who_woman': 0, 'alone_True': 1},
        {'pclass': 3, 'age': 10, 'sibsp': 2, 'parch': 1, 'class': 0, 'adult_male': 0, 'sex_male': 0, 'embarked_Q': 0, 'embarked_S': 1, 'embark_town_Queenstown': 0, 'embark_town_Southampton': 1, 'who_man': 0, 'who_woman': 0, 'alone_True': 0},
        {'pclass': 2, 'age': 30, 'sibsp': 1, 'parch': 0, 'class': 1, 'adult_male': 0, 'sex_male': 0, 'embarked_Q': 0, 'embarked_S': 0, 'embark_town_Queenstown': 0, 'embark_town_Southampton': 0, 'who_man': 0, 'who_woman': 1, 'alone_True': 0}
    ]
    
    df_inputs = pd.DataFrame(inputs)
    preds = model.predict(df_inputs)
    probs = model.predict_proba(df_inputs)[:, 1]


    sys_prompt = "You are a data science explainer. Output ONLY valid JSON matching this schema: prediction_label, confidence_level, top_reason, second_reason, next_step. ALL VALUES MUST BE STRINGS, EVEN NUMBERS. No markdown blocks."
    
    user_template = "Features: {f}. Prediction: {pred}. Probability of survival: {prob}. Explain this."


    for i in range(3):
        print(f"\n--- Record {i+1} ---")
        user_prompt = user_template.format(f=inputs[i], pred=preds[i], prob=probs[i])
        
        print(f"Features: {inputs[i]}")
        print(f"Pred: {preds[i]}, Prob: {probs[i]:.4f}")
        
        for temp in [0.0, 0.7]:
            print(f"\nCalling LLM with temp={temp}")
            resp = call_llm(sys_prompt, user_prompt, temperature=temp)
            
            if resp is None:
                print("LLM returned None. using fallback.")
                final_json = fallback
                status = "fail"
            else:
                resp = resp.strip()
                if resp.startswith("```json"):
                    resp = resp[7:]
                if resp.endswith("```"):
                    resp = resp[:-3]
                    
                try:
                    parsed = json.loads(resp)
                    try:
                        jsonschema.validate(parsed, schema)
                        final_json = parsed
                        status = "pass"
                    except jsonschema.ValidationError as e:
                        print("Schema Error:", e.message)
                        final_json = fallback
                        status = "fail"
                except json.JSONDecodeError as e:
                    print("JSON Decode Error:", e)
                    final_json = fallback
                    status = "fail"
                    
            print(f"Status: {status}")
            print(f"Output: {final_json}")


if __name__ == "__main__":
    main()
