"""
Token cost benchmark.

The claim under test: driving PotatOpt costs a language model far fewer tokens
than writing the same predictive-maintenance pipeline by hand, because the
boilerplate - splitting, imputing, encoding, scaling, weighting, threshold
search, metric assembly - is already inside the library.

Run it:

    python benchmarks/token_cost.py

Uses the `tiktoken` tokeniser when it is installed and falls back to a
characters-per-token estimate otherwise. The output says which one it used, so a
number copied out of this script is never mistaken for something it is not.

The three snippets below must stay in step with the README quick start. They are
what a competent engineer would write for the same task, not strawmen.
"""

BASELINE = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

df = pd.read_csv("sensors.csv").dropna(subset=["failure"])
X = df.drop(columns=["failure"])
y = df["failure"]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

num_imp = SimpleImputer(strategy="median").fit(X_train[num_cols])
cat_imp = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols]) if cat_cols else None
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
scaler = StandardScaler()

def prep(frame, fit=False):
    out = frame.copy()
    out[num_cols] = num_imp.transform(out[num_cols])
    if cat_cols:
        out[cat_cols] = cat_imp.transform(out[cat_cols])
        out[cat_cols] = enc.fit_transform(out[cat_cols]) if fit else enc.transform(out[cat_cols])
    out[num_cols] = scaler.fit_transform(out[num_cols]) if fit else scaler.transform(out[num_cols])
    return out.astype(np.float32)

X_train_p, X_val_p, X_test_p = prep(X_train, fit=True), prep(X_val), prep(X_test)

weights = compute_sample_weight("balanced", y_train)
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
model.fit(X_train_p, y_train, sample_weight=weights)

proba_val = model.predict_proba(X_val_p)[:, 1]
best_t, best_cost = 0.5, float("inf")
for t in np.arange(0.05, 0.95, 0.05):
    pred = (proba_val >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    cost = fn * 500 + fp * 150 + tp * 20
    if cost < best_cost:
        best_cost, best_t = cost, t

proba_test = model.predict_proba(X_test_p)[:, 1]
y_pred = (proba_test >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f1_score(y_test, y_pred), roc_auc_score(y_test, proba_test),
      average_precision_score(y_test, proba_test), fn * 500 + fp * 150 + tp * 20)
'''

STEP_BY_STEP = '''import pandas as pd
from potatopt import PotatOptEngine, split_data_three_way, inspect_data, audit_data_quality

df = pd.read_csv("sensors.csv")
report = inspect_data(df, target_col="failure")
audit = audit_data_quality(df, target_col="failure")
X_train, X_val, X_test, y_train, y_val, y_test = split_data_three_way(
    df, target_col="failure", val_size=0.2, test_size=0.2
)
engine = PotatOptEngine(task="auto", time_budget=30, cost_sensitive_weighting=True, n_jobs=-1)
engine.fit(X_train, y_train)
engine.optimize_threshold(X_val, y_val, cost_scrap=500.0, cost_fa=150.0, cost_insp=20.0)
metrics = engine.evaluate(X_test, y_test)
coq = engine.calculate_cost_of_quality(X_test, y_test, cost_scrap=500, cost_fa=150, cost_insp=20)
engine.save("model.pkl")
print(metrics["f1"], coq["cost_savings"])
'''

FACADE = '''import potatopt as po

result = po.auto_analyze("sensors.csv", target="failure",
                         cost_scrap=500, cost_fa=150, cost_insp=20)
print(result["metrics"]["f1"], result["cost"]["cost_savings"])
'''

CHARS_PER_TOKEN = 3.6

VARIANTS = [
    ("scikit-learn by hand", BASELINE),
    ("PotatOpt step by step", STEP_BY_STEP),
    ("PotatOpt auto_analyze", FACADE),
]


def count_tokens(source):
    """Return (token_count, method_name) for one snippet."""
    try:
        import tiktoken
    except ImportError:
        return round(len(source) / CHARS_PER_TOKEN), f"estimate ({CHARS_PER_TOKEN} chars/token)"
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(source)), "tiktoken cl100k_base"


def measure():
    """Return one row of results per variant."""
    rows = []
    method = None
    for name, source in VARIANTS:
        tokens, method = count_tokens(source)
        code_lines = len([line for line in source.strip().splitlines() if line.strip()])
        rows.append({"variant": name, "lines": code_lines, "chars": len(source), "tokens": tokens})
    baseline_tokens = rows[0]["tokens"]
    for row in rows:
        row["saving_pct"] = round(100 * (1 - row["tokens"] / baseline_tokens), 1)
    return rows, method


def main():
    rows, method = measure()
    print(f"Token counting method: {method}")
    print()
    print(f"{'variant':<24}{'lines':>7}{'chars':>8}{'tokens':>9}{'saving':>9}")
    print("-" * 57)
    for row in rows:
        saving = "-" if row["saving_pct"] == 0 else f"{row['saving_pct']}%"
        print(f"{row['variant']:<24}{row['lines']:>7}{row['chars']:>8}{row['tokens']:>9}{saving:>9}")
    print()
    print("Same task in every row: train a predictive-maintenance classifier,")
    print("tune the decision threshold on cost, report metrics and money saved.")
    print("The baseline does not include AutoML search, memory downcasting, drift")
    print("detection, the data-quality gate, SHAP, or a signed model file - all of")
    print("which the PotatOpt rows get without another line of code.")


if __name__ == "__main__":
    main()
