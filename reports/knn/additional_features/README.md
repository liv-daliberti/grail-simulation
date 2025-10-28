# Additional Text Features


Overview of the supplementary text columns appended to the viewer prompt alongside the prompt builder output.

## Next-Video Pipeline

### Sweep Configurations

No sweep metrics were supplied.

### Final Evaluations

No final evaluation metrics were supplied.

## Opinion Regression

### Sweep Configurations

| Feature space | Study | Configuration | Extra text fields |
| --- | --- | --- | --- |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 3 – Minimum Wage (YouGov) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |

### Final Evaluations

No opinion metrics were recorded.

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: none (defaults only).
