# GRPO Next-Video Baseline

- **Overall accuracy:** 0.757 on 300 eligible slates out of 300 processed.
- **Parsed rate:** 0.983
- **Formatted rate:** 0.983

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 175 | 175 | 0.977 | 1.000 | 1.000 |
| minimum_wage | 125 | 125 | 0.448 | 0.960 | 0.960 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 175 | 175 | 0.977 | 1.000 | 1.000 |
| study2 | 45 | 45 | 0.444 | 1.000 | 1.000 |
| study3 | 80 | 80 | 0.450 | 0.938 | 0.938 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
