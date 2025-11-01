#!/usr/bin/env python
"""Shared default extra text fields for viewer prompts.

Both the KNN and XGBoost CLIs expose a default list of extended text fields
to append to the base prompt document. Centralising the constant here avoids
duplicating the long CSV header across modules.
"""

DEFAULT_EXTENDED_TEXT_FIELDS = (
    "pid1,pid2,ideo1,ideo2,pol_interest,religpew,educ,employ,child18,inputstate,"
    "freq_youtube,youtube_time,newsint,q31,participant_study,slate_source,"
    "minwage_text_w2,minwage_text_w1,mw_support_w2,mw_support_w1,minwage15_w2,"
    "minwage15_w1,mw_index_w2,mw_index_w1,gun_importance,gun_index,gun_enthusiasm,"
    "gun_identity"
)

__all__ = [
    "DEFAULT_EXTENDED_TEXT_FIELDS",
]
