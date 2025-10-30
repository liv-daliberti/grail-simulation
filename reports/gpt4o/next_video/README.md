# GPT-4o Next-Video Baseline

- **Selected configuration:** `temp0_tok500_tp1` (temperature=0.00, top_p=1.00, max_tokens=500)
- **Accuracy:** 0.263 on 2419 eligible slates out of 2419 processed.
- **Parsed rate:** 1.000  **Formatted rate:** 1.000

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.261 | 1.000 | 1.000 |
| minimum_wage | 1871 | 1871 | 0.263 | 1.000 | 1.000 |

### Highlights

- Highest accuracy: minimum_wage (0.263, eligible 1871).
- Lowest accuracy: gun_control (0.261, eligible 548).

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.261 | 1.000 | 1.000 |
| study2 | 671 | 671 | 0.286 | 1.000 | 1.000 |
| study3 | 1200 | 1200 | 0.251 | 1.000 | 1.000 |

### Highlights

- Highest accuracy: study2 (0.286, eligible 671).
- Lowest accuracy: study3 (0.251, eligible 1200).

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
