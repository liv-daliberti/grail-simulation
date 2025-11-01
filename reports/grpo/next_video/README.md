# GRPO Next-Video Baseline

- **Overall accuracy:** 0.737 on 650 eligible slates out of 650 processed.
- **Parsed rate:** 0.977
- **Formatted rate:** 0.975

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 375 | 375 | 0.963 | 0.997 | 0.997 |
| minimum_wage | 275 | 275 | 0.429 | 0.949 | 0.945 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 375 | 375 | 0.963 | 0.997 | 0.997 |
| study2 | 90 | 90 | 0.478 | 0.978 | 0.978 |
| study3 | 185 | 185 | 0.405 | 0.935 | 0.930 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
