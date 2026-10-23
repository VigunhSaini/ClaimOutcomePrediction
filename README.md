# Insurance Claim Outcome Predictor

This is a small machine learning project that predicts what will happen to an insurance claim:
- **Paid** (paid in full)
- **Partially_Paid** (insurance only paid part of it)
- **Denied** (insurance refused to pay)

It uses past claim information (like who the insurance company is, what procedure was done, and how much money was billed) to make the prediction, and serves it through a small web API.

---

## Project Files

- `generate_data.py`: Creates `claims.csv` with 450 realistic claims.
- `features.py`: Builds new helpful features without cheating (no data leakage).
- `train.py`: Splits the data, trains the model, checks how well it works, and saves it to `model.joblib`.
- `app.py`: A simple Flask web API that takes a claim and returns the predicted outcome.
- `bonus_appeal.py`: A helper script that finds appeal tips for different denial reasons.
- `test_api.py`: Tests to make sure the API works and handles bad inputs properly.
- `requirements.txt`: The list of Python libraries needed to run this.
- `README.md`: This guide.

---

## How to Run It

### 1. Install libraries
```bash
pip install -r requirements.txt
```

### 2. Make the data
```bash
python generate_data.py
```
This creates `claims.csv` with 450 claims (about 70% Paid, 20% Partially Paid, and 10% Denied).

### 3. Train the model
```bash
python train.py
```
This trains the model, prints out the test scores, and saves `model.joblib`.

### 4. Run tests
```bash
pytest -v test_api.py
```
This runs 8 automated checks to make sure the API handles good and bad requests correctly.

### 5. Start the API
```bash
python app.py
```

Now you can test it by sending a claim to `http://127.0.0.1:5000/predict`.

#### Example request:
```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"payer\": \"Aetna\", \"provider\": \"Dr. Rao\", \"cpt_code\": \"99213\", \"diagnosis_code\": \"M54.5\", \"billed_amount\": 220.0, \"patient_age\": 45}"
```

#### Example response:
```json
{
  "predicted_status": "Denied",
  "probabilities": {
    "Paid": 0.14,
    "Partially_Paid": 0.37,
    "Denied": 0.49
  }
}
```
*(This claim gets predicted as Denied because the normal price for code 99213 is $120, so billing $220 is way higher than usual).*

---

## What Features the Model Uses

The model looks at:
1. `payer`: Which insurance company it is (e.g. Aetna, UnitedHC).
2. `cpt_code`: The code for the medical visit/procedure.
3. `billed_amount`: How much money the doctor charged.
4. `billed_to_cpt_mean_ratio`: We created this feature. It divides the billed amount by the average price for that procedure. If someone bills $250 for a $100 visit, this number is 2.5, which is a big red flag.
5. `payer_denial_rate`: How often this specific insurance company denies claims historically.
6. `payer_partial_rate`: How often this insurance company only pays part of the bill.

### Why we dropped Doctor Name, Diagnosis, and Age:
We tested the model with and without doctor names, diagnosis codes, and patient age. The score stayed the same, meaning they didn't help the model make better guesses. We left them out of the model to keep things simple. (The API still takes them so it matches the test rules, but the model ignores them).

---

## How Well Does the Model Work?

We tested the model on 90 claims it had never seen before:

| Outcome | Precision | Recall | F1-Score | Number of Claims in Test |
| :--- | :---: | :---: | :---: | :---: |
| **Paid** | 80% | 64% | 0.71 | 63 |
| **Partially_Paid** | 33% | 35% | 0.34 | 17 |
| **Denied** | 32% | 70% | 0.44 | 10 |
| **Overall Accuracy** | — | — | **59%** | 90 |

### What do Precision and Recall mean here?
- **Recall (for Denied)**: Out of all the claims that were actually Denied, how many did we catch? The model caught **7 out of 10 (70%)**.
- **Precision (for Denied)**: When the model yelled "Denied!", was it actually right? It was right about **32%** of the time (it flagged 22 claims as Denied, and 7 of them really were).
- **Why is accuracy only 59%?**
  If a model simply guessed "Paid" for every single claim, it would get 70% accuracy because most claims are Paid. But it would catch **zero** denials! We told our model to care equally about denials, so it is willing to make some false alarms in order to catch the real denials.

### Confusion Matrix (what really happened on the 90 test claims):
- Real Paid claims (63): 40 predicted Paid, 11 predicted Partially Paid, 12 predicted Denied.
- Real Partially Paid claims (17): 8 predicted Paid, 6 predicted Partially Paid, 3 predicted Denied.
- Real Denied claims (10): 2 predicted Paid, 1 predicted Partially Paid, **7 caught as Denied**.

---

## Which Class Was the Hardest?

**Partially_Paid was the hardest to predict.** 

Why? Because partially paid claims are right in the middle between Paid and Denied. In our data, unless a claim had a really high bill or came from Aetna with a specific procedure, it usually got paid normally. Because most normal claims get paid in full, the model often had trouble telling a normal claim apart from a partially paid one.

---

## What is Data Leakage and How Did We Stop It?

Data leakage is like accidentally giving a student the answer sheet before they take a test.

If you calculate "how often does Aetna deny claims" using the whole dataset before splitting it into training and testing piles, the training pile gets information about the test answers. When you test the model, it looks like a genius, but in the real world it falls flat.

**How we stopped it:**
We wrote a custom feature builder in `features.py`. It only learns the denial rates from the **training data**. When it sees test data (or brand new claims in the API), it only looks up what it learned during training. If it sees a brand new insurance company it has never seen before, it uses a safe average instead of crashing.

---

## Assumptions and Shortcuts

1. **Fake Data**: We generated 450 fake claims with Python. Real hospital claims are messier and have thousands of rows.
2. **Small Test Set**: There were only 10 denied claims in our test set. If the model misses just one claim, the recall drops by 10%. So the test numbers can jump around easily.
3. **Simple Settings**: We used sensible basic settings for the Random Forest model instead of spending hours searching for tiny tweaks.
4. **We Updated the Data Generator Once**: Our very first version of the fake data was too random, so the model couldn't learn anything. We changed the generator once to create clearer rules (like UnitedHC denying high bills more often).

---

## Bonus: Appeal Tips & Stopping AI from Making Things Up

In `bonus_appeal.py`, we wrote a small function that takes a reason like "missing_auth" and returns helpful advice on how to appeal it.

If we ever hooked this up to an AI (like ChatGPT) to write the actual appeal letter, here is how to stop it from making up fake facts (hallucinating):
1. **Give it only the real facts**: Tell the AI to use only the patient's real bill and the advice snippet we gave it.
2. **Tell it to say "I don't know"**: If it doesn't have a date or policy number, it must write "Information not provided" instead of inventing a fake number.
3. **Use a fixed template**: Fill in patient names, dates, and dollar amounts using normal code, not the AI.
4. **Turn randomness down**: Set temperature to 0 so the AI doesn't get creative.
5. **Always have a human check it**: A real person should read the letter before it gets mailed to the insurance company.
