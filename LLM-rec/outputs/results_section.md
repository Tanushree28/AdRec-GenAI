# Results

## 5. Results

### 5.1 Regression Performance

Table 1 reports regression metrics on the KuaiRec `big_matrix` validation set (seed=42 split, watch-ratio clipped at 5.0). The LLM-Enhanced MLP achieves an MSE of 0.4942, a RMSE of 0.7030, and an MAE of 0.4375, compared to 0.5041, 0.7100, and 0.4443 for the Baseline MLP — improvements of **1.96%, 0.99%, and 1.53%**, respectively. Although the absolute reductions are modest, they are consistent across all three metrics, indicating that frozen sentence-transformer item embeddings provide a stable, additive signal beyond collaborative-filtering ID features alone. This is consistent with findings in the broader literature on semantic augmentation of recommendation models, where small but reliable gains over strong ID-only baselines are considered meaningful given the inherent noise in engagement signals such as watch-ratio.

### 5.2 Ranking Performance

Figure 1 presents a grouped bar chart of ranking metrics at K=10. The LLM-Enhanced MLP outperforms the Baseline on all four ranking metrics: Precision@10 improves from 0.6862 to 0.7006 (+2.10%), Recall@10 from 0.2001 to 0.2050 (+2.45%), NDCG@10 from 0.8762 to 0.8849 (+0.99%), and HitRate@10 from 0.9980 to 0.9985. Gains are consistent across cut-offs K ∈ {5, 10, 20} (Table 1), which rules out threshold sensitivity as a confounding factor. The notably high absolute NDCG values (>0.87 for both models) reflect the high engagement density of the KuaiRec dataset, where the majority of users have at least one highly relevant item in their validation interactions. Within this ceiling-constrained setting, a uniform ranking improvement across all cut-offs further validates the utility of semantic item representations.

### 5.3 Ablation Study

Figure 2 shows the best-epoch validation MSE for all four ablation variants trained under identical conditions (seed=42, early stopping with patience=5). Item-LLM achieves the lowest validation loss (0.5018), followed by Full-LLM (0.5032), User-LLM (0.5033), and Baseline (0.5041). Three observations are notable. First, every LLM-augmented variant outperforms the ID-only Baseline, confirming that pre-trained language representations carry genuine preference signal. Second, **item-side semantics contribute more than user-side semantics**: Item-LLM (0.5018) outperforms User-LLM (0.5033), suggesting that video caption embeddings are richer and more discriminative than the behaviorally derived user profile texts used here. Third, Full-LLM does not dominate Item-LLM despite having access to both embedding sources, which may indicate mild redundancy between the user LLM embedding and the learnable user ID embedding, or that the user text construction (activity features + top-3 categories) requires further refinement. These findings motivate future work on richer user-side text and cross-modal fusion strategies.

### 5.4 Recommendation Diversity

Figure 3 reports diversity metrics computed over per-user top-10 recommendation lists. The LLM-Enhanced model achieves a slightly higher long-tail coverage (0.0078 vs. 0.0075), indicating a marginally greater tendency to surface items with fewer historical interactions — a desirable property that reduces popularity bias. Category entropy, however, is marginally lower for LLM-Enhanced (2.124 vs. 2.191 bits). This trade-off is consistent with item-embedding-guided models concentrating recommendations within semantically coherent clusters: the model discovers deeper within a narrower set of relevant categories rather than spreading uniformly across all categories. Overall, both models exhibit comparable diversity profiles, and neither exhibits catastrophic popularity collapse, suggesting that embedding augmentation does not significantly harm recommendation breadth.

---

## Metrics Summary Table

| Metric | Baseline MLP | LLM-Enhanced MLP | Improvement |
|--------|-------------|-----------------|-------------|
| MSE | 0.5041 | **0.4942** | -1.96% |
| RMSE | 0.7100 | **0.7030** | -0.99% |
| MAE | 0.4443 | **0.4375** | -1.53% |
| Precision@5 | 0.7210 | **0.7360** | +2.08% |
| Recall@5 | 0.1086 | **0.1118** | +2.95% |
| NDCG@5 | 0.8781 | **0.8881** | +1.14% |
| HitRate@5 | 0.9792 | **0.9851** | +0.60% |
| Precision@10 | 0.6862 | **0.7006** | +2.10% |
| Recall@10 | 0.2001 | **0.2050** | +2.45% |
| NDCG@10 | 0.8762 | **0.8849** | +0.99% |
| HitRate@10 | 0.9980 | **0.9985** | +0.05% |
| Precision@20 | 0.6306 | **0.6419** | +1.79% |
| Recall@20 | 0.3410 | **0.3466** | +1.64% |
| NDCG@20 | 0.8708 | **0.8792** | +0.97% |
| HitRate@20 | 1.0000 | 1.0000 | 0.00% |
| Category Entropy | **2.1912** | 2.1243 | -3.05% |
| Longtail Coverage | 0.0075 | **0.0078** | +4.00% |

---

## Ablation Study Summary

| Variant | Best Val MSE | Best Epoch | Total Epochs |
|---------|-------------|-----------|-------------|
| Baseline (ID-only) | 0.5041 | 9 | 14 |
| **Item-LLM** | **0.5018** | 20 | 25 |
| User-LLM | 0.5033 | 8 | 13 |
| Full-LLM | 0.5032 | 7 | 12 |

---

## Figure Captions

**Figure 1.** Grouped bar chart comparing Baseline MLP and LLM-Enhanced MLP on regression metrics (MSE, RMSE, MAE; lower is better) and ranking metrics at K=10 (Precision, Recall, NDCG, HitRate; higher is better) on the KuaiRec `big_matrix` validation set.

**Figure 2.** Ablation study: best-epoch validation MSE for four model variants — Baseline (ID-only), Item-LLM (+ item sentence embeddings), User-LLM (+ user sentence embeddings), and Full-LLM (+ both). Lower is better. All LLM variants outperform the Baseline.

**Figure 3.** Diversity metrics for top-10 recommendation lists: category entropy (Shannon entropy of recommended item categories) and long-tail coverage (fraction of recommended items below the 20th-percentile popularity threshold). Higher is better for both.

---

## Key Takeaways

- **Core claim holds**: LLM item embeddings consistently improve both regression accuracy and ranking quality — the improvement is not an artefact of a single metric or cut-off.
- **Item semantics > user semantics** in the current setup; the gap suggests user profile text construction is a bottleneck worth investigating.
- **Diminishing returns from stacking**: Full-LLM does not surpass Item-LLM, pointing to redundancy rather than complementarity between user ID embeddings and user LLM embeddings.
- **Diversity is preserved**: Embedding augmentation does not collapse recommendations toward popular items; long-tail coverage actually improves slightly.
- **Ceiling effect warning**: KuaiRec's high engagement density inflates absolute ranking scores; readers should interpret relative gains rather than absolute values.
