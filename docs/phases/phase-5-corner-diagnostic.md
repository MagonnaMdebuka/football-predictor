# Phase 5 Corner Diagnostic

Walk-forward backtest on 760 held-out Premier League matches (2024-25 and
2025-26). Corner model: NB2 with team attack/defence effects, fitted per date.

## 1. Predicted vs Observed Variance

|                | Predicted (model) | Observed |
|----------------|-------------------|----------|
| Var(home)      | 8.315             | 8.768    |
| Var(away)      | 6.544             | 8.368    |
| Cov(H,A)       | **0 (assumed)**   | **-2.897** |
| Var(total)     | **14.859**        | **11.349** |
| Mean total     | 10.317            | 10.147   |

The model assumes home and away corners are independent. In reality they are
negatively correlated: the team that dominates possession wins more corners and
concedes fewer.

**Correlation between home and away corners within a match:**

- Pearson:  r = -0.3378, p < 0.0001
- Spearman: r = -0.3049, p < 0.0001

The independence assumption overestimates total variance by 31%
(14.9 vs 11.3). The missing -2.9 covariance term accounts for the entire gap.
A convolution of two independent NB2 marginals produces a total distribution
that is too wide, hedging O/U probabilities toward 50/50 instead of making
sharp predictions.

## 2. Decile Calibration

760 matches sorted by predicted total corners and split into deciles.

| Decile       |  N | Pred Mean | Actual Mean | Pred Range      | Actual Std |
|--------------|---:|----------:|------------:|-----------------|------------|
| 1 (lowest)   | 76 |      8.63 |        9.00 | [6.9, 9.2]      |       3.16 |
| 2            | 76 |      9.41 |        9.39 | [9.2, 9.6]      |       3.34 |
| 3            | 76 |      9.71 |       10.08 | [9.6, 9.8]      |       3.14 |
| 4            | 76 |      9.95 |       10.86 | [9.8, 10.1]     |       3.43 |
| 5            | 76 |     10.17 |        9.42 | [10.1, 10.3]    |       2.93 |
| 6            | 76 |     10.39 |       10.63 | [10.3, 10.5]    |       2.91 |
| 7            | 76 |     10.63 |       10.13 | [10.5, 10.7]    |       3.15 |
| 8            | 76 |     10.92 |       10.59 | [10.7, 11.1]    |       3.76 |
| 9            | 76 |     11.32 |       10.54 | [11.1, 11.6]    |       3.40 |
| 10 (highest) | 76 |     12.04 |       10.83 | [11.6, 13.8]    |       3.76 |

- Decile mean correlation: r = 0.7461, p = 0.0132
- Regression slope (actual on predicted): 0.506
- Actual range across deciles: 1.83 corners
- Match-level pred vs actual total: Pearson r = 0.1340, p = 0.0002

The curve slopes but at half-strength. The model predicts a 3.4-corner spread
across deciles (8.6 to 12.0) but actual only rises 1.8 (9.0 to 10.8). Team
effects are reaching the totals — the ranking is broadly correct — but the
model overspreads predictions. It overestimates totals for strong-corner
fixtures and underestimates for weak ones, consistent with the independence
assumption inflating variance.

## 3. Three-Way Brier Comparison

Gate lines 8.5-12.5 (publishable subset, excluding lopsided extremes 7.5 and
13.5).

| Line | Model Brier | Simple Brier | Base Brier | Model vs Base | Simple vs Base |
|------|-------------|-------------|------------|---------------|----------------|
|  8.5 | 0.219957    | 0.218077    | 0.216550   | +1.6%         | +0.7%          |
|  9.5 | 0.246463    | 0.243833    | 0.243558   | +1.2%         | +0.1%          |
| 10.5 | 0.249777    | 0.247914    | 0.248442   | +0.5%         | -0.2%          |
| 11.5 | 0.217946    | 0.217952    | 0.218447   | -0.2%         | -0.2%          |
| 12.5 | 0.181204    | 0.179733    | 0.181439   | -0.1%         | -0.9%          |
| **Gate mean** | **0.223069** | **0.221502** | **0.221687** | **+0.6%** | **-0.1%** |

- **Model**: NB2 convolution with team attack/defence effects (walk-forward).
- **Simple**: predicted total = (team_for + opponent_against) / 2 + mirror,
  Poisson CDF for O/U probabilities. Static training set.
- **Base**: constant over-rate per line from held-out actuals.

The simple rate-based model ties base-rate (-0.1%) while the NB2 model loses
(+0.6%). The simple model is better than the NB2 model on every line. Team
effects contain signal; the NB2 convolution destroys it.

## Diagnosis

Corner team effects are real (attack std = 0.14, defence std = 0.18, decile
r = 0.75). The problem is not that corners are unpredictable. The problem is
the **independence assumption in the NB2 convolution**. Home and away corners
have r = -0.34, meaning the convolved total distribution is 31% too wide. This
produces hedged O/U probabilities that lose to naive over-rates.

A model that accounts for the negative covariance (bivariate NB2, direct
total-corners model, or copula correction) should recover the signal that the
current architecture destroys.
