# GPT-4o Hyper-parameter Sweep

The table below captures validation accuracy on eligible slates plus formatting/parse rates for each temperature/top-p/max-token configuration. The selected configuration is marked with ✓.

| Config | Temperature | Top-p | Max tokens | Accuracy ↑ | Parsed ↑ | Formatted ↑ | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `temp0_tok500_tp1` | 0.00 | 1.00 | 500 | 0.263 | 1.000 | 1.000 | ✓ |
| `temp0p2_tok500_tp1` | 0.20 | 1.00 | 500 | 0.262 | 1.000 | 1.000 |  |
| `temp0p4_tok500_tp1` | 0.40 | 1.00 | 500 | 0.253 | 1.000 | 1.000 |  |
