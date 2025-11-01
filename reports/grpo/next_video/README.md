# GRPO Next-Video Baseline

- **Overall accuracy:** 0.660 on 1,223 eligible slates out of 1,223 processed.
- **Parsed rate:** 0.969
- **Formatted rate:** 0.964

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.962 | 0.998 | 0.998 |
| minimum_wage | 675 | 675 | 0.415 | 0.945 | 0.936 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.962 | 0.998 | 0.998 |
| study2 | 242 | 242 | 0.372 | 0.959 | 0.950 |
| study3 | 433 | 433 | 0.439 | 0.938 | 0.928 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
