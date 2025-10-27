# Additional Text Features


Overview of the supplementary text columns appended to the viewer prompt alongside the prompt builder output.

## Next-Video Pipeline

### Sweep Configurations

| Feature space | Study | Configuration | Extra text fields |
| --- | --- | --- | --- |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 3 – Minimum Wage (YouGov) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-l2_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-l2_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 3 – Minimum Wage (YouGov) | metric-l2_text-viewerprofile_statetext | `viewer_profile`, `state_text` |

### Final Evaluations

| Feature space | Study | Extra text fields |
| --- | --- | --- |
| tfidf | Study 1 – Gun Control (MTurk) | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | `viewer_profile`, `state_text` |

## Opinion Regression

### Sweep Configurations

| Feature space | Study | Configuration | Extra text fields |
| --- | --- | --- | --- |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 3 – Minimum Wage (YouGov) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-l2_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-l2_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 3 – Minimum Wage (YouGov) | metric-l2_text-viewerprofile_statetext | `viewer_profile`, `state_text` |

### Final Evaluations

No opinion metrics were recorded.

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: none (defaults only).
