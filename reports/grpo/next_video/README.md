# GRPO Next-Video Baseline

- **Overall accuracy:** 0.675 on 1,166 eligible slates out of 1,166 processed.
- **Parsed rate:** 0.970
- **Formatted rate:** 0.965

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.962 | 0.998 | 0.998 |
| minimum_wage | 618 | 618 | 0.421 | 0.945 | 0.935 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.962 | 0.998 | 0.998 |
| study2 | 214 | 214 | 0.393 | 0.953 | 0.944 |
| study3 | 404 | 404 | 0.436 | 0.941 | 0.931 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
