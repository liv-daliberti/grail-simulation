# Visualized Recommendation Trees

This report collects four Graphviz exports produced by the `visualize_recommendation_trees` tooling in `src/visualization/recommendation_tree_viz.py`. The assets previously lived under `docs/batch_sessions`; they now reside in `reports/visualized_recommendation_trees/figures` so the trees can ship alongside other analyst-facing summaries.

To regenerate any figure, rerun the visualizer with the desired session identifier, for example:

```bash
python -m src.visualization.recommendation_tree_viz --session-id <session_id> --output-dir reports/visualized_recommendation_trees/figures
```

## Gun Control Sessions

### grail_session_gun_control_1
![Gun control session 1](figures/grail_session_gun_control_1.svg)
- Start: `Do We Need Stricter Gun Control? – The People Speak`
- Clicked path: moved into `How to Create a Gun-Free America in 5 Easy Steps`, then into a televised debate clip (`Piers Morgan Argues With Pro-Gun Campaigner About Orlando Shooting`)
- Notable slate: recommendations mix argumentative commentary with satirical takes

### grail_session_gun_control_2
![Gun control session 2](figures/grail_session_gun_control_2.svg)
- Start: `Common arguments for gun control, shot down`
- Clicked path: dives into `Florida's New Gun Control Explained in 6 Minutes – The Legal Brief!`, then `“GUN CONFISCATION BECOMES A REALITY IN ILLINOIS!”`
- Notable slate: subsequent options emphasize red-flag laws and national policy pushes (`20 States Looking To Pass Red Flag Laws`, `NJ Attempts Confiscation…`)

## Minimum Wage Sessions

### grail_session_minimum_wage_1
![Minimum wage session 1](figures/grail_session_minimum_wage_1.svg)
- Start: `Seattle's $15 Minimum Wage Experiment is Working`
- Clicked path: transitions to the explainer `Price Floors: The Minimum Wage`, followed by `The 5 Biggest Myths Republicans Use to Avoid Raising the Minimum Wage`, and explores `Who Does a $15 Minimum Wage Help?`
- Notable slate: juxtaposes pro–minimum wage narratives with counterarguments on inflation and business viability

### grail_session_minimum_wage_2
![Minimum wage session 2](figures/grail_session_minimum_wage_2.svg)
- Start: `Raise The Minimum Wage — Robert Reich & MoveOn.org`
- Clicked path: pivots to opposing content (`Stossel: Minimum Wage Hurts Beginners`), then circles back through `What the US gets wrong about minimum wage`, culminating at the NowThis myth-busting feature before surfacing additional labor-market clips
- Notable slate: highlights alternating exposure to pro- and anti-minimum wage messaging within a single viewing sequence

