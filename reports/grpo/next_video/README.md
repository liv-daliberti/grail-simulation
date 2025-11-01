# GRPO Next-Video Baseline

- **Overall accuracy:** 0.789 on 175 eligible slates out of 175 processed.
- **Parsed rate:** 0.983
- **Formatted rate:** 0.983

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 100 | 100 | 0.990 | 1.000 | 1.000 |
| minimum_wage | 75 | 75 | 0.520 | 0.960 | 0.960 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 100 | 100 | 0.990 | 1.000 | 1.000 |
| study2 | 20 | 20 | 0.400 | 1.000 | 1.000 |
| study3 | 55 | 55 | 0.564 | 0.945 | 0.945 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
