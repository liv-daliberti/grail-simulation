# GRPO Next-Video Baseline

- **Overall accuracy:** 0.673 on 1,173 eligible slates out of 1,173 processed.
- **Parsed rate:** 0.970
- **Formatted rate:** 0.965

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.962 | 0.998 | 0.998 |
| minimum_wage | 625 | 625 | 0.421 | 0.946 | 0.936 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.962 | 0.998 | 0.998 |
| study2 | 221 | 221 | 0.394 | 0.955 | 0.946 |
| study3 | 404 | 404 | 0.436 | 0.941 | 0.931 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
