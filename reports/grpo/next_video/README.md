# GRPO Next-Video Baseline

- **Overall accuracy:** 0.554 on 2,123 eligible slates out of 2,123 processed.
- **Parsed rate:** 0.956
- **Formatted rate:** 0.949

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.962 | 0.998 | 0.998 |
| minimum_wage | 1,575 | 1,575 | 0.413 | 0.941 | 0.931 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.962 | 0.998 | 0.998 |
| study2 | 550 | 550 | 0.360 | 0.938 | 0.924 |
| study3 | 1,025 | 1,025 | 0.441 | 0.942 | 0.936 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
