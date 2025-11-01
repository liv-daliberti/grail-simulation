# GRPO Next-Video Baseline

- **Overall accuracy:** 0.584 on 1,748 eligible slates out of 1,748 processed.
- **Parsed rate:** 0.958
- **Formatted rate:** 0.953

## Accuracy by Issue

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| gun_control | 548 | 548 | 0.962 | 0.998 | 0.998 |
| minimum_wage | 1,200 | 1,200 | 0.411 | 0.940 | 0.932 |

## Accuracy by Participant Study

| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| study1 | 548 | 548 | 0.962 | 0.998 | 0.998 |
| study2 | 423 | 423 | 0.357 | 0.941 | 0.929 |
| study3 | 777 | 777 | 0.440 | 0.940 | 0.934 |

### Notes

Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.
