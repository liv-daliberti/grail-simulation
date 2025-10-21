# KNN Next-Video Baseline

This report summarises the existing slate-ranking KNN model that predicts the next video a viewer will click from the study slate.

- Dataset: `data/cleaned_grail`
- Split: validation (548 gun-control slates, 1 871 minimum-wage slates)
- Metric: accuracy on eligible slates (gold index present)

## TF-IDF Feature Space

| Issue | Accuracy ↑ | Best k | Most-frequent baseline ↑ |
| --- | ---: | ---: | ---: |
| Gun control | 0.894 | 2 | 0.540 |
| Minimum wage | 0.306 | 3 | 0.439 |

- Plots: `tfidf/elbow_gun_control.png`, `tfidf/elbow_minimum_wage.png`

## Word2Vec Feature Space

| Issue | Accuracy ↑ | Best k | Most-frequent baseline ↑ |
| --- | ---: | ---: | ---: |
| Gun control | 0.870 | 2 | 0.540 |
| Minimum wage | 0.288 | 3 | 0.439 |

- Plots: `word2vec/elbow_gun_control.png`, `word2vec/elbow_minimum_wage.png`

## Observations

- Gun-control slates remain a strong fit for KNN with accuracy > 0.87 across feature spaces and a best k of 2.
- Minimum-wage slates are challenging: both feature spaces underperform a naïve most-frequent baseline, suggesting feature sparsity or noisier prompts for that issue.
- Accuracy declines steadily as k grows beyond the low single digits, reinforcing the need for tight neighbourhoods in slate prediction.
