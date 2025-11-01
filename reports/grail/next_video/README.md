# GRAIL Next-Video Baseline

- **Overall accuracy:** 0.733 on 1,023 eligible slates out of 1,023 processed.
- **Parsed rate:** 0.999
- **Formatted rate:** 0.999

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.964 | 1.000 | 1.000 |
| minimum_wage | 475 | 475 | 0.467 | 0.998 | 0.998 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.964 | 1.000 | 1.000 |
| study2 | 167 | 167 | 0.419 | 0.994 | 0.994 |
| study3 | 308 | 308 | 0.494 | 1.000 | 1.000 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
