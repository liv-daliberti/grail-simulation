# GRPO Next-Video Baseline

- **Overall accuracy:** 0.566 on 1,923 eligible slates out of 1,923 processed.
- **Parsed rate:** 0.957
- **Formatted rate:** 0.951

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.962 | 0.998 | 0.998 |
| minimum_wage | 1,375 | 1,375 | 0.409 | 0.941 | 0.932 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.962 | 0.998 | 0.998 |
| study2 | 482 | 482 | 0.349 | 0.944 | 0.929 |
| study3 | 893 | 893 | 0.441 | 0.940 | 0.933 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
