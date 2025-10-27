# Additional Text Features


Overview of the supplementary text columns appended to the viewer prompt alongside the prompt builder output.

## Next-Video Pipeline

### Sweep Configurations

| Feature space | Study | Configuration | Extra text fields |
| --- | --- | --- | --- |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext | `viewer_profile`, `state_text` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_ideo1 | `viewer_profile`, `state_text`, `ideo1` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext_ideo1 | `viewer_profile`, `state_text`, `ideo1` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_ideo2 | `viewer_profile`, `state_text`, `ideo2` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext_ideo2 | `viewer_profile`, `state_text`, `ideo2` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_polinterest | `viewer_profile`, `state_text`, `pol_interest` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext_polinterest | `viewer_profile`, `state_text`, `pol_interest` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_religpew | `viewer_profile`, `state_text`, `religpew` |
| tfidf | Study 2 – Minimum Wage (MTurk) | metric-cosine_text-viewerprofile_statetext_religpew | `viewer_profile`, `state_text`, `religpew` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_freqyoutube | `viewer_profile`, `state_text`, `freq_youtube` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_youtubetime | `viewer_profile`, `state_text`, `youtube_time` |
| tfidf | Study 1 – Gun Control (MTurk) | metric-cosine_text-viewerprofile_statetext_newsint | `viewer_profile`, `state_text`, `newsint` |

### Final Evaluations

No final evaluation metrics were supplied.

## Opinion Regression

### Sweep Configurations

No opinion sweep metrics were supplied.

### Final Evaluations

No opinion metrics were recorded.

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: `freq_youtube`, `ideo1`, `ideo2`, `newsint`, `pol_interest`, `religpew`, `youtube_time`
