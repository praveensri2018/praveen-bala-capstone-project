# Part 4: LLM-Powered Feature

### How to Run
1. Install requirements using `pip install -r ../requirements.txt` (if not already installed). You will also need the `python-dotenv` and `requests` packages.
2. Create a file named `.env` in this `part4` directory.
3. Add your API key inside `.env` exactly like this:
   `LLM_API_KEY=your-actual-api-key-here`
4. Run the script: `python maincodepart4.py`

---

I chose **Track C: Model Prediction Explanation Pipeline**.


### Prompts Used

**System Prompt:**
`You are a data science explainer. Output ONLY valid JSON matching this schema: prediction_label, confidence_level, top_reason, second_reason, next_step. ALL VALUES MUST BE STRINGS, EVEN NUMBERS. No markdown blocks.`

**User Prompt Template:**
`Features: {f}. Prediction: {pred}. Probability of survival: {prob}. Explain this.`


### Why Temperature = 0?
We use `temperature=0` because it forces the LLM to always choose the most likely next token, making the output deterministic and highly predictable. This is essential for structured data extraction where we need exactly formatted JSON. Higher temperatures (like 0.7) sample from a broader distribution of tokens, which introduces creative variability but drastically increases the risk of the model hallucinating invalid JSON formatting or rambling.


### Temperature A/B Comparison
| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Record 3 (Prob: 0.9450) | `{"prediction_label": "1", "confidence_level": "0.945016", "top_reason": "The individual is a female, which historically had a higher survival rate.", "second_reason": "The individual is in a second-class cabin, which had better survival rates compared to third class.", "next_step": "Consider additional factors such as the individual's health and the circumstances of the disaster."}` | `{"prediction_label": "1", "confidence_level": "0.945016", "top_reason": "The individual is a woman, which historically has a higher survival rate.", "second_reason": "The individual is in a second-class cabin, which has better survival rates compared to third class.", "next_step": "Consider additional factors such as lifeboat accessibility and the individual's health status."}` | Temp 0 is very direct. Temp 0.7 used slightly different phrasing ("woman" vs "female" and "lifeboat accessibility" vs "circumstances of the disaster"). |

*Note: As expected, temp=0.7 introduces variability and slight phrasing changes which is dangerous if we strictly enforce enum values in the JSON schema.*


### PII Guardrail
Before calling the LLM, I run a regex check for emails and phone numbers.
- **Input with PII**: `"my email is test@test.com"` -> Output: `Input blocked: PII detected.`
- **Clean Input**: `"Reply with only the word: hello"` -> Proceeded normally to LLM.


### JSON Schema Validation
I expected the LLM to output a JSON dictionary with these 5 string fields:
- `prediction_label`
- `confidence_level`
- `top_reason`
- `second_reason`
- `next_step`

If the JSON doesn't match, `jsonschema.validate` catches it and I return a fallback dictionary with `None` values instead of crashing.


### End-to-End Demonstration Table

| Feature Input | Predicted Class | Probability | Explanation JSON | Valid JSON | Pass/Block |
|---|---|---|---|---|---|
| pclass=1, age=40, adult_male=1, class=2, sex_male=1... | 0 | 0.2910 | `{"prediction_label": "0", "confidence_level": "0.291022", "top_reason": "The individual is a male in first class, which typically has lower survival rates.", "second_reason": "The individual is traveling alone, which may decrease chances of survival compared to those with companions.", "next_step": "Consider additional factors such as the specific circumstances of the voyage to refine the prediction."}` | pass | Pass |
| pclass=3, age=10, sibsp=2, parch=1, class=0, sex_male=0... | 0 | 0.3990 | `{"prediction_label": "0", "confidence_level": "0.398986", "top_reason": "The individual is a child, but traveling with multiple siblings and a parent may not significantly increase survival chances.", "second_reason": "The individual is in third class, which historically had lower survival rates compared to first and second class.", "next_step": "Consider additional factors such as the specific circumstances of the voyage and historical data on survival rates."}` | pass | Pass |
| pclass=2, age=30, sibsp=1, parch=0, class=1, who_woman=1... | 1 | 0.9450 | `{"prediction_label": "1", "confidence_level": "0.945016", "top_reason": "The individual is a female, which historically had a higher survival rate.", "second_reason": "The individual is in a second-class cabin, which had better survival rates compared to third class.", "next_step": "Consider additional factors such as the individual's health and the circumstances of the disaster."}` | pass | Pass |
