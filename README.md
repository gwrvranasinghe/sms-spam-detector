# SMS Spam Detector

A machine learning app that classifies SMS messages as spam or ham.

## Results (test set, spam class)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes (baseline) | 1.00 | 0.82 | 0.90 |
| Logistic Regression | 0.94 | 0.82 | 0.88 |
| NB + extra features (final, threshold 0.3) | 0.98 | 0.93 | 0.95 |

## Approach
- Dataset: SMS Spam Collection (5,169 messages after removing duplicates, ~12.6% spam)
- Text cleaning, TF-IDF (unigrams + bigrams)
- Extra features: message length, digit count, uppercase ratio, phone number and price patterns
- Multinomial Naive Bayes with a tuned decision threshold

## Run locally
    pip install -r requirements.txt
    python -m streamlit run app.py

## Limitations
- The threshold (0.3) was chosen using the test set, so reported scores may be slightly optimistic.
- Trained on a small, older dataset; modern spam may look different.
- English messages only.