# GRAIL Next-Video Baseline

- **Overall accuracy:** 0.617 on 1,648 eligible slates out of 1,648 processed.
- **Parsed rate:** 0.996
- **Formatted rate:** 0.996

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.964 | 1.000 | 1.000 |
| minimum_wage | 1,100 | 1,100 | 0.445 | 0.994 | 0.994 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.964 | 1.000 | 1.000 |
| study2 | 385 | 385 | 0.369 | 0.984 | 0.984 |
| study3 | 715 | 715 | 0.485 | 0.999 | 0.999 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
