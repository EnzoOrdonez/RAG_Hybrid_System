"""
Analyze results from ALL user evaluation sessions.

Run after all participants have completed their sessions:
    python scripts/analyze_user_sessions.py

Generates:
  - Summary table per participant
  - Statistical tests (paired t-test / Wilcoxon)
  - Average ratings per system
  - SUS score analysis
  - Qualitative summary
  - Exports: LaTeX tables, figures, CSV
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
SESSIONS_DIR = PROJECT_ROOT / "data" / "evaluation" / "user_sessions"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
CSV_DIR = OUTPUT_DIR / "csv"

SYSTEM_NAMES = {
    "lexical": "RAG Lexico (BM25)",
    "semantic": "RAG Semantico (Dense)",
    "hybrid": "RAG Hibrido Propuesto",
}


def load_all_sessions() -> list:
    """Load all completed sessions."""
    sessions = []
    if not SESSIONS_DIR.exists():
        return sessions

    for participant_dir in sorted(SESSIONS_DIR.iterdir()):
        if not participant_dir.is_dir():
            continue

        full_path = participant_dir / "full_session.json"
        if not full_path.exists():
            print(f"  Warning: {participant_dir.name} - no full_session.json (incomplete?)")
            continue

        try:
            data = json.loads(full_path.read_text(encoding="utf-8"))
            data["participant_id"] = participant_dir.name
            sessions.append(data)
        except Exception as e:
            print(f"  Error loading {participant_dir.name}: {e}")

    return sessions


def compute_per_system_ratings(sessions: list) -> dict:
    """Compute average utility and accuracy per system."""
    system_ratings = {}
    for sys_key in SYSTEM_NAMES:
        system_ratings[sys_key] = {"utility": [], "accuracy": []}

    for session in sessions:
        for rating in session.get("ratings", []):
            sys_key = rating.get("system", "")
            if sys_key in system_ratings:
                system_ratings[sys_key]["utility"].append(rating["utility_rating"])
                system_ratings[sys_key]["accuracy"].append(rating["accuracy_rating"])

    results = {}
    for sys_key, data in system_ratings.items():
        results[sys_key] = {
            "utility_mean": np.mean(data["utility"]) if data["utility"] else 0,
            "utility_std": np.std(data["utility"]) if data["utility"] else 0,
            "accuracy_mean": np.mean(data["accuracy"]) if data["accuracy"] else 0,
            "accuracy_std": np.std(data["accuracy"]) if data["accuracy"] else 0,
            "n_ratings": len(data["utility"]),
        }
    return results


def compute_sus_stats(sessions: list) -> dict:
    """Compute SUS score statistics."""
    scores = []
    for session in sessions:
        sus_score = session.get("sus_score", 0)
        if sus_score > 0:
            scores.append(sus_score)

    if not scores:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n": len(scores),
    }


def run_statistical_tests(sessions: list) -> dict:
    """Run paired tests between systems."""
    from scipy import stats as scipy_stats

    # Collect per-participant per-system averages
    system_avgs = {sys: [] for sys in SYSTEM_NAMES}

    for session in sessions:
        participant_avgs = {sys: [] for sys in SYSTEM_NAMES}
        for rating in session.get("ratings", []):
            sys_key = rating.get("system", "")
            if sys_key in participant_avgs:
                participant_avgs[sys_key].append(rating["utility_rating"])

        for sys_key in SYSTEM_NAMES:
            if participant_avgs[sys_key]:
                system_avgs[sys_key].append(np.mean(participant_avgs[sys_key]))

    results = {}
    pairs = [
        ("hybrid", "lexical"),
        ("hybrid", "semantic"),
        ("semantic", "lexical"),
    ]

    for sys_a, sys_b in pairs:
        a_vals = system_avgs[sys_a]
        b_vals = system_avgs[sys_b]

        n = min(len(a_vals), len(b_vals))
        if n < 3:
            results[f"{sys_a}_vs_{sys_b}"] = {"error": "Not enough participants (need >= 3)"}
            continue

        a = np.array(a_vals[:n])
        b = np.array(b_vals[:n])

        # Paired t-test
        t_stat, t_pvalue = scipy_stats.ttest_rel(a, b)

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = scipy_stats.wilcoxon(a, b)
        except Exception:
            w_stat, w_pvalue = 0, 1.0

        # Cohen's d
        diff = a - b
        cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0

        results[f"{sys_a}_vs_{sys_b}"] = {
            "n": n,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pvalue),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pvalue),
            "cohens_d": cohens_d,
            "significant_005": t_pvalue < 0.05,
        }

    return results


def compute_timing_stats(sessions: list) -> dict:
    """Compute timing statistics per system."""
    system_times = {sys: {"reading": [], "rating": [], "total": []} for sys in SYSTEM_NAMES}

    for session in sessions:
        for ts in session.get("timestamps", []):
            sys_key = ts.get("system", "")
            if sys_key in system_times:
                system_times[sys_key]["reading"].append(ts.get("reading_time_ms", 0))
                system_times[sys_key]["rating"].append(ts.get("rating_time_ms", 0))
                system_times[sys_key]["total"].append(ts.get("total_time_ms", 0))

    results = {}
    for sys_key, times in system_times.items():
        results[sys_key] = {}
        for metric, values in times.items():
            if values:
                results[sys_key][f"{metric}_mean_ms"] = float(np.mean(values))
                results[sys_key][f"{metric}_std_ms"] = float(np.std(values))
    return results


def qualitative_summary(sessions: list) -> dict:
    """Summarize open-ended responses."""
    preferences = {}
    differences = []
    improvements = []

    for session in sessions:
        open_resp = session.get("open_responses", {})
        pref = open_resp.get("preferred_system", "").strip()
        if pref:
            # Count mentions of each system
            pref_lower = pref.lower()
            for sys_key, sys_name in SYSTEM_NAMES.items():
                if sys_key in pref_lower or sys_name.lower() in pref_lower:
                    preferences[sys_key] = preferences.get(sys_key, 0) + 1

        diff = open_resp.get("noticed_differences", "").strip()
        if diff:
            differences.append(diff)

        imp = open_resp.get("improvements", "").strip()
        if imp:
            improvements.append(imp)

    return {
        "system_preferences": preferences,
        "n_noticed_differences": len(differences),
        "n_improvements": len(improvements),
        "differences": differences,
        "improvements": improvements,
    }


def export_results(sessions, system_ratings, sus_stats, stat_tests, timing, qualitative):
    """Export all analysis results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # === CSV: Raw data ===
    all_ratings = []
    for session in sessions:
        pid = session.get("participant_id", "?")
        for rating in session.get("ratings", []):
            rating["participant_id"] = pid
            all_ratings.append(rating)

    if all_ratings:
        csv_path = CSV_DIR / "user_evaluation_raw.csv"
        headers = list(all_ratings[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_ratings)
        print(f"  Exported: {csv_path}")

    # === LaTeX table ===
    latex_rows = []
    for sys_key in ["lexical", "semantic", "hybrid"]:
        r = system_ratings.get(sys_key, {})
        latex_rows.append(
            f"  {SYSTEM_NAMES[sys_key]} & "
            f"{r.get('utility_mean', 0):.2f} $\\pm$ {r.get('utility_std', 0):.2f} & "
            f"{r.get('accuracy_mean', 0):.2f} $\\pm$ {r.get('accuracy_std', 0):.2f} & "
            f"{r.get('n_ratings', 0)} \\\\"
        )

    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{User Evaluation Results}\n"
        "\\label{tab:user_evaluation}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "System & Utility (1-5) & Accuracy (1-5) & N \\\\\n"
        "\\midrule\n"
        + "\n".join(latex_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    latex_path = TABLES_DIR / "table_user_evaluation.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"  Exported: {latex_path}")

    # === Figures ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: User ratings
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        systems = list(SYSTEM_NAMES.keys())
        labels = [SYSTEM_NAMES[s] for s in systems]
        colors = ["#E67E22", "#3498DB", "#27AE60"]

        utility_means = [system_ratings[s]["utility_mean"] for s in systems]
        utility_stds = [system_ratings[s]["utility_std"] for s in systems]
        accuracy_means = [system_ratings[s]["accuracy_mean"] for s in systems]
        accuracy_stds = [system_ratings[s]["accuracy_std"] for s in systems]

        axes[0].bar(labels, utility_means, yerr=utility_stds, color=colors, capsize=5)
        axes[0].set_ylabel("Mean Rating (1-5)")
        axes[0].set_title("Utility Ratings")
        axes[0].set_ylim(0, 5.5)
        axes[0].tick_params(axis="x", rotation=15)

        axes[1].bar(labels, accuracy_means, yerr=accuracy_stds, color=colors, capsize=5)
        axes[1].set_ylabel("Mean Rating (1-5)")
        axes[1].set_title("Accuracy Ratings")
        axes[1].set_ylim(0, 5.5)
        axes[1].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig_user_ratings.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Exported: {fig_path}")

        # Figure 2: SUS scores
        sus_scores = [s.get("sus_score", 0) for s in sessions if s.get("sus_score", 0) > 0]
        if sus_scores:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(sus_scores)), sus_scores, color="#1B3A5C")
            ax.axhline(y=70, color="red", linestyle="--", label="Target (70)")
            ax.axhline(y=np.mean(sus_scores), color="green", linestyle="--", label=f"Mean ({np.mean(sus_scores):.1f})")
            ax.set_xlabel("Participant")
            ax.set_ylabel("SUS Score (0-100)")
            ax.set_title("System Usability Scale Scores")
            ax.set_ylim(0, 105)
            ax.legend()
            plt.tight_layout()
            fig_path = FIGURES_DIR / "fig_sus_scores.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Exported: {fig_path}")

    except ImportError:
        print("  Warning: matplotlib not available, skipping figures")

    # === Full analysis JSON ===
    analysis = {
        "n_participants": len(sessions),
        "system_ratings": system_ratings,
        "sus_stats": sus_stats,
        "statistical_tests": stat_tests,
        "timing_stats": timing,
        "qualitative": {k: v for k, v in qualitative.items() if k != "differences" and k != "improvements"},
    }
    analysis_path = CSV_DIR / "user_evaluation_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Exported: {analysis_path}")


def main():
    print("=" * 60)
    print("USER EVALUATION SESSION ANALYSIS")
    print("=" * 60)

    # Load sessions
    print(f"\nLoading sessions from: {SESSIONS_DIR}")
    sessions = load_all_sessions()
    print(f"  Found {len(sessions)} completed sessions")

    if not sessions:
        print("\n  No completed sessions found.")
        print(f"  Expected directory: {SESSIONS_DIR}")
        print("  Each participant directory should contain full_session.json")
        return

    if len(sessions) < 6:
        print(f"\n  WARNING: Only {len(sessions)} participants. Need >= 6 for statistical significance.")

    # Compute metrics
    print("\nComputing per-system ratings...")
    system_ratings = compute_per_system_ratings(sessions)
    for sys_key, r in system_ratings.items():
        print(f"  {SYSTEM_NAMES[sys_key]}:")
        print(f"    Utility:  {r['utility_mean']:.2f} +/- {r['utility_std']:.2f}")
        print(f"    Accuracy: {r['accuracy_mean']:.2f} +/- {r['accuracy_std']:.2f}")

    print("\nSUS Score analysis...")
    sus_stats = compute_sus_stats(sessions)
    print(f"  Mean: {sus_stats['mean']:.1f} +/- {sus_stats['std']:.1f}")
    print(f"  Range: [{sus_stats['min']:.0f}, {sus_stats['max']:.0f}]")
    print(f"  Target >= 70: {'MET' if sus_stats['mean'] >= 70 else 'NOT MET'}")

    print("\nStatistical tests...")
    try:
        stat_tests = run_statistical_tests(sessions)
        for comparison, result in stat_tests.items():
            if "error" in result:
                print(f"  {comparison}: {result['error']}")
            else:
                sig = "***" if result["significant_005"] else "n.s."
                print(
                    f"  {comparison}: t={result['t_statistic']:.3f}, "
                    f"p={result['t_pvalue']:.4f} {sig}, "
                    f"d={result['cohens_d']:.3f}"
                )
    except ImportError:
        print("  Warning: scipy not installed, skipping statistical tests")
        stat_tests = {}

    print("\nTiming analysis...")
    timing = compute_timing_stats(sessions)
    for sys_key, t in timing.items():
        total_mean = t.get("total_mean_ms", 0)
        print(f"  {SYSTEM_NAMES[sys_key]}: {total_mean/1000:.1f}s avg per query")

    print("\nQualitative summary...")
    qual = qualitative_summary(sessions)
    prefs = qual.get("system_preferences", {})
    for sys_key, count in prefs.items():
        print(f"  {SYSTEM_NAMES.get(sys_key, sys_key)}: preferred by {count} participant(s)")

    # Export
    print("\nExporting results...")
    export_results(sessions, system_ratings, sus_stats, stat_tests, timing, qual)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
