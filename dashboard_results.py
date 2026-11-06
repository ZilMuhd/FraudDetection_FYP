"""
Reported results shown by the dashboard (copied from fyp2_progress_log.md).
These are the numbers in the dissertation. If a result changes, change it here
and in the log together.
"""

PAYSIM = {
    "dataset": {"raw_rows": 6_362_620, "rows": 2_770_409, "fraud": 8_213, "fraud_rate": 0.002965,
                "accounts": 3_277_509, "types": "TRANSFER and CASH_OUT only",
                "removed": "oldbalanceOrg, newbalanceOrig (leakage); account IDs"},
    "louvain": {"communities": 507_105, "median_size": 4, "largest": 90,
                "accounts_in_2plus_fraud": 44, "max_fraud_per_account": 2},
    "centrality": [  # fraud vs legitimate, mean (median)
        {"feature": "dest_pageRank", "fraud": "0.877 (0.533)", "legit": "1.630 (1.298)"},
        {"feature": "dest_degree", "fraud": "5.70 (3.0)", "legit": "11.60 (9.0)"},
        {"feature": "origin_pageRank", "fraud": "0.150 (0.150)", "legit": "0.150 (0.150)"},
        {"feature": "origin_degree", "fraud": "1.00 (1.0)", "legit": "1.00 (1.0)"},
    ],
    "iterations": [
        {"model": "Iteration 1", "change": "SMOTE 1.0, no community size", "precision": 0.1384, "recall": 0.6713, "f1": 0.2296, "pr_auc": 0.5122, "roc_auc": 0.9395},
        {"model": "Iteration 2", "change": "SMOTE 0.3", "precision": 0.2388, "recall": 0.5959, "f1": 0.3410, "pr_auc": 0.4830, "roc_auc": 0.9421},
        {"model": "Iteration 3 (final, random split)", "change": "+ community size", "precision": 0.2402, "recall": 0.5916, "f1": 0.3417, "pr_auc": 0.4787, "roc_auc": 0.9410},
    ],
    "comparisons": [
        {"id": "R1", "experiment": "XGBoost, all features (Iteration 3)", "precision": 0.2402, "recall": 0.5916, "pr_auc": 0.4787, "false_alarms": 3074},
        {"id": "R2", "experiment": "XGBoost, no graph features", "precision": 0.0644, "recall": 0.6981, "pr_auc": 0.4230, "false_alarms": 16666},
        {"id": "R3", "experiment": "XGBoost, leaking balance columns kept", "precision": 0.6099, "recall": 0.9811, "pr_auc": 0.9726, "false_alarms": 1031},
        {"id": "R4", "experiment": "Random Forest, all features", "precision": 0.2073, "recall": 0.5648, "pr_auc": 0.4023, "false_alarms": 3549},
        {"id": "R5", "experiment": "LightGBM, all features", "precision": 0.2312, "recall": 0.5746, "pr_auc": 0.4467, "false_alarms": 3139},
    ],
    "future_journey": [
        {"version": "Graph features from the full timeline (leak)", "pr_auc": 0.4487, "finding": "Features already contained future transactions"},
        {"version": "Leak-free, raw step (T1)", "pr_auc": 0.2283, "finding": "Future steps were all higher than any step seen in training"},
        {"version": "Leak-free, hour of day (T3)", "pr_auc": 0.4859, "finding": "Graph hurt on known receivers: train-serve skew"},
        {"version": "Rolling window (final, RW-G)", "pr_auc": 0.5556, "finding": "Training and test features built the same way"},
    ],
    "known_new_t3t4": [
        {"receiver": "Known", "transactions": 39406, "fraud": 524, "T3 graph": 0.3025, "T4 no graph": 0.5075},
        {"receiver": "New", "transactions": 11880, "fraud": 1130, "T3 graph": 0.6438, "T4 no graph": 0.5703},
    ],
    "rolling": {
        "periods": [
            {"period": "History", "steps": "1-475", "role": "Builds the graph only", "transactions": 2_633_880, "fraud": 5_315},
            {"period": "Training window", "steps": "476-594", "role": "Training; features looked up from history", "transactions": 85_243, "fraud": 1_244},
            {"period": "Prediction", "steps": "595-743", "role": "Unseen test; features looked up from steps 1-594", "transactions": 51_286, "fraud": 1_654},
        ],
        "models": [
            {"model": "RW-G (graph + amount, hour, type)", "precision": 0.4327, "recall": 0.5671, "f1": 0.4908, "pr_auc": 0.5556, "roc_auc": 0.9035, "false_alarms": 1230, "caught": 938},
            {"model": "RW-NG (amount, hour, type only)", "precision": 0.1855, "recall": 0.4504, "f1": 0.2628, "pr_auc": 0.3642, "roc_auc": float("nan"), "false_alarms": 3271, "caught": 745},
        ],
        "by_receiver": [
            {"receiver": "Known", "RW-G": 0.4604, "RW-NG": 0.4290},
            {"receiver": "New", "RW-G": 0.5853, "RW-NG": 0.4551},
        ],
        "importance": {"type_is_transfer": 0.317, "dest_pageRank": 0.226, "dest_degree": 0.178, "hour": 0.131,
                       "dest_communitySize": 0.100, "amount": 0.041},
        "chance": 0.0323,
    },
    "robustness": [  # PR-AUC mean +/- std over 5 seeds, rolling window
        {"algorithm": "XGBoost", "boundary": 475, "graph": 0.5538, "graph_std": 0.0029, "no_graph": 0.3633, "no_graph_std": 0.0041},
        {"algorithm": "LightGBM", "boundary": 475, "graph": 0.5820, "graph_std": 0.0012, "no_graph": 0.3871, "no_graph_std": 0.0031},
        {"algorithm": "Random Forest", "boundary": 475, "graph": 0.5333, "graph_std": 0.0067, "no_graph": 0.3614, "no_graph_std": 0.0038},
        {"algorithm": "XGBoost", "boundary": 400, "graph": 0.5521, "graph_std": 0.0027, "no_graph": 0.3724, "no_graph_std": 0.0025},
        {"algorithm": "LightGBM", "boundary": 400, "graph": 0.5732, "graph_std": 0.0021, "no_graph": 0.4050, "no_graph_std": 0.0031},
        {"algorithm": "Random Forest", "boundary": 400, "graph": 0.5353, "graph_std": 0.0060, "no_graph": 0.3707, "no_graph_std": 0.0023},
    ],
    "thresholds": [  # chosen on validation (steps 560-594), applied once to the test period, boundary 475
        {"features": "Graph", "rule": "Default", "threshold": 0.500, "precision": 0.4327, "recall": 0.5671, "false_alarms": 1230},
        {"features": "Graph", "rule": "Recall 0.75 on validation", "threshold": 0.089, "precision": 0.2105, "recall": 0.7473, "false_alarms": 4636},
        {"features": "Graph", "rule": "Best F1 on validation", "threshold": 0.989, "precision": 0.8915, "recall": 0.2932, "false_alarms": 59},
        {"features": "No graph", "rule": "Recall 0.75 on validation", "threshold": 0.054, "precision": 0.0559, "recall": 0.7225, "false_alarms": 20171},
    ],
}

SIMULATION = {
    "dataset": {"normal_accounts": 50_000, "normal_tx": 200_000, "rings": 40, "mules_per_ring": "5-12",
                "events_per_ring": "15-40", "victims_per_ring": 60,
                "train_rows": 204_545, "train_fraud": 4_545, "pred_rows": 204_150, "pred_fraud": 4_150},
    "louvain": [
        {"community": 7397, "type": "Fraud ring", "fraud_tx": 167, "total_tx": 167, "density": 1.0},
        {"community": 20238, "type": "Fraud ring", "fraud_tx": 161, "total_tx": 161, "density": 1.0},
        {"community": 20187, "type": "Normal", "fraud_tx": 0, "total_tx": 10822, "density": 0.0},
        {"community": 3464, "type": "Normal", "fraud_tx": 0, "total_tx": 10763, "density": 0.0},
    ],
    "models": [
        {"model": "Full features (incl. community size)", "precision": 0.9978, "recall": 1.0000, "f1": 0.9989, "pr_auc": 1.0000, "roc_auc": 1.0000},
        {"model": "Without community size (final)", "precision": 0.9667, "recall": 0.9593, "f1": 0.9630, "pr_auc": 0.9941, "roc_auc": 0.9998},
        {"model": "Final model on separate prediction batch", "precision": 0.9564, "recall": 0.9260, "f1": 0.9410, "pr_auc": 0.9773, "roc_auc": 0.9984},
    ],
    "importance_full": {"dest_communitySize": 0.726, "origin_communitySize": 0.274},
    "importance_final": {"origin_degree": 0.461, "dest_degree": 0.356, "dest_pageRank": 0.076, "origin_pageRank": 0.070,
                         "type_is_transfer": 0.031, "amount": 0.003, "step": 0.003},
}

NOISY = {
    "settings": [
        {"noise source": "Victims who are normal customers", "clean": "0%", "low": "25%", "medium": "50%", "high": "90%", "basis": "Lopez-Rojas et al. (2016)"},
        {"noise source": "Everyday transactions per mule", "clean": "0", "low": "5", "medium": "20", "high": "50", "basis": "Europol (n.d., 2019); Dou et al. (2020)"},
        {"noise source": "Laundering shapes", "clean": "chain", "low": "+ fan-in, fan-out", "medium": "+ cycle, scatter-gather", "high": "all 5", "basis": "Altman et al. (2023)"},
        {"noise source": "Fraud events with missing labels", "clean": "0%", "low": "5%", "medium": "15%", "high": "30%", "basis": "Altman et al. (2023); Dal Pozzolo et al. (2018)"},
        {"noise source": "Mules replaced per week", "clean": "0%", "low": "5%", "medium": "15%", "high": "30%", "basis": "FCA (2026); Europol (2019). Rate: design choice"},
    ],
    # Louvain share = share of all fraud inside communities that are >= 50% fraud,
    # MEAN of 5 reloads (louvain_stability.py); min/max = range over the 5 runs.
    "levels": {
        "clean":  {"louvain_share": 0.8726, "louvain_min": 0.8533, "louvain_max": 0.8960, "G": 1.0000, "G-noCS": 0.9102, "NG": 0.0100, "chance": 0.0097, "tx": 503_922, "fraud": 3_922},
        "low":    {"louvain_share": 0.1789, "louvain_min": 0.1218, "louvain_max": 0.2585, "G": 0.7541, "G-noCS": 0.7755, "NG": 0.0125, "chance": 0.0133, "tx": 508_592, "fraud": 5_344},
        "medium": {"louvain_share": 0.1258, "louvain_min": 0.0000, "louvain_max": 0.2302, "G": 0.4075, "G-noCS": 0.3023, "NG": 0.0160, "chance": 0.0141, "tx": 519_515, "fraud": 6_120},
        "high":   {"louvain_share": 0.0657, "louvain_min": 0.0000, "louvain_max": 0.1458, "G": 0.0842, "G-noCS": 0.3660, "NG": 0.0141, "chance": 0.0131, "tx": 539_597, "fraud": 6_296},
    },
    "ofat": [
        {"run": "Clean", "louvain_share": 0.8726, "louvain_min": 0.8533, "louvain_max": 0.8960, "G-noCS": 0.910, "G": 1.000},
        {"run": "Victims only", "louvain_share": 0.1633, "louvain_min": 0.0484, "louvain_max": 0.3247, "G-noCS": 0.839, "G": 0.896},
        {"run": "Mule everyday transactions only", "louvain_share": 0.0813, "louvain_min": 0.0000, "louvain_max": 0.2116, "G-noCS": 0.698, "G": 0.690},
        {"run": "Shapes only", "louvain_share": 0.8544, "louvain_min": 0.8280, "louvain_max": 0.8870, "G-noCS": 0.775, "G": 0.226},
        {"run": "Missing labels only", "louvain_share": 0.8679, "louvain_min": 0.8503, "louvain_max": 0.8950, "G-noCS": 0.890, "G": 1.000},
        {"run": "Mule turnover only", "louvain_share": 0.8862, "louvain_min": 0.8672, "louvain_max": 0.9047, "G-noCS": 0.961, "G": 0.998},
        {"run": "High noise, fan-in only", "louvain_share": 0.0000, "louvain_min": 0.0000, "louvain_max": 0.0000, "G-noCS": 0.178, "G": 0.059},
        {"run": "High noise, scatter-gather only", "louvain_share": 0.3186, "louvain_min": 0.0536, "louvain_max": 0.5144, "G-noCS": 0.493, "G": 0.503},
    ],
    "community_size_check": [
        {"level": "Clean", "changed_2x": 0.057, "normal_median": "5,453 -> 5,165", "normal_in_fraud_range": "0% -> 0%", "G_false_alarms": 0},
        {"level": "Low", "changed_2x": 0.721, "normal_median": "1,901 -> 6,554", "normal_in_fraud_range": "68.5% -> 14.4%", "G_false_alarms": 1833},
        {"level": "Medium", "changed_2x": 0.432, "normal_median": "2,762 -> 5,105", "normal_in_fraud_range": "85.8% -> 3.8%", "G_false_alarms": 1219},
        {"level": "High", "changed_2x": 0.694, "normal_median": "2,012 -> 5,890", "normal_in_fraud_range": "85.3% -> 100%", "G_false_alarms": 73860},
    ],
}

OBJECTIVES = [
    {"objective": "i) Model transactions as a graph and use Louvain to isolate high-risk clusters",
     "evidence": "Simulation: fraud rings 100% fraud, normal communities 0%. PaySim: no clusters, because PaySim never reuses mule accounts. Noisy simulation: camouflage cuts isolation from about 87% to 7-18% on average, and makes Louvain unstable (0-32% across reloads of the same data); tightly connected laundering shapes survive in some runs only."},
    {"objective": "ii) Hybrid pipeline fusing PageRank and Degree Centrality with transaction features for XGBoost",
     "evidence": "Built end to end; graph features are top contributors in every final model."},
    {"objective": "iii) Reduce false positives while keeping fraud detection recall",
     "evidence": "PaySim future data: 62% fewer false alarms and 26% more fraud caught than without graph features; at 75% recall, about 4 times fewer false alarms. Held across 3 algorithms, 5 seeds and two boundaries."},
]
