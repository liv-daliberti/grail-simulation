# Prompt Builder Data Assets

This folder stores static data files referenced by the prompt builder.

- `video_stats.json` – aggregate engagement metadata (views, comments, etc.)
  keyed by video ID. The prompt builder uses these statistics to annotate slate
  options with consistent “Engagement” summaries.

When updating the JSON, keep the schema stable (`video_id` keys with numeric
stats) so existing helpers continue to parse it. Large derived datasets should
live in `data/` instead; the prompt builder only keeps lightweight artefacts
here.
