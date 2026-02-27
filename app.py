import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def fmt_p(p: float) -> str:
    return "<0.01" if p < 0.01 else f"{p:.4f}"


def fmt_prob(p: float) -> str:
    """Format a Bayesian probability, capping 100% as >99.9%."""
    if p >= 0.999:
        return ">99.9%"
    if p <= 0.001:
        return "<0.1%"
    return f"{p:.1%}"


def int_input(label: str, default: int, *, key: str) -> int | None:
    raw = st.text_input(label, value=str(default), key=key)
    cleaned = raw.strip().replace(",", "")
    try:
        v = int(cleaned)
    except ValueError:
        st.error(f"{label}: enter a whole number.")
        return None
    if v < 0:
        st.error(f"{label}: cannot be negative.")
        return None
    return v


def float_input(label: str, default: float, *, key: str) -> float | None:
    raw = st.text_input(label, value=str(default), key=key)
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("£", "")
    try:
        v = float(cleaned)
    except ValueError:
        st.error(f"{label}: enter a valid number.")
        return None
    if v < 0:
        st.error(f"{label}: cannot be negative.")
        return None
    return v


# ---------------------------------------------------------------------------
# Frequentist statistical tests
# ---------------------------------------------------------------------------

def z_test_proportions(
    count_a: int, n_a: int, count_b: int, n_b: int
) -> dict:
    """Two-proportion z-test (two-sided). A = control, B = variant."""
    p_a = count_a / n_a
    p_b = count_b / n_b
    p_pooled = (count_a + count_b) / (n_a + n_b)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_a = np.sqrt(p_a * (1 - p_a) / n_a)
    se_b = np.sqrt(p_b * (1 - p_b) / n_b)
    ci_a = (p_a - 1.96 * se_a, p_a + 1.96 * se_a)
    ci_b = (p_b - 1.96 * se_b, p_b + 1.96 * se_b)

    lift = (p_b / p_a - 1) if p_a > 0 else 0.0

    return {
        "rate_a": p_a,
        "rate_b": p_b,
        "ci_a": ci_a,
        "ci_b": ci_b,
        "z": z,
        "p_value": p_value,
        "lift": lift,
    }


def t_test_aov(
    rev_a: float, orders_a: int, rev_b: float, orders_b: int, cv: float = 1.0
) -> dict:
    """Welch's t-test for AOV (Average Order Value) using summary statistics.

    cv (coefficient of variation) is used to estimate the standard deviation
    from the mean since we only have aggregate revenue. A cv of 1.0 is a
    reasonable conservative default for e-commerce order values.
    """
    mean_a = rev_a / orders_a if orders_a > 0 else 0.0
    mean_b = rev_b / orders_b if orders_b > 0 else 0.0

    sd_a = max(mean_a * cv, 0.01)
    sd_b = max(mean_b * cv, 0.01)

    if orders_a < 2 or orders_b < 2:
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "ci_a": (mean_a, mean_a),
            "ci_b": (mean_b, mean_b),
            "t": 0.0,
            "p_value": 1.0,
            "lift": 0.0,
        }

    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=mean_a, std1=sd_a, nobs1=orders_a,
        mean2=mean_b, std2=sd_b, nobs2=orders_b,
        equal_var=False,
    )

    se_a = sd_a / np.sqrt(orders_a)
    se_b = sd_b / np.sqrt(orders_b)
    ci_a = (mean_a - 1.96 * se_a, mean_a + 1.96 * se_a)
    ci_b = (mean_b - 1.96 * se_b, mean_b + 1.96 * se_b)

    lift = (mean_b / mean_a - 1) if mean_a > 0 else 0.0

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "ci_a": ci_a,
        "ci_b": ci_b,
        "t": float(t_stat),
        "p_value": float(p_value),
        "lift": lift,
    }


# ---------------------------------------------------------------------------
# Bayesian engine
# ---------------------------------------------------------------------------

def bayesian_ab_test(
    successes_a: int, total_a: int,
    successes_b: int, total_b: int,
    prior_alpha: float = 1.0, prior_beta: float = 1.0,
    n_sim: int = 10_000,
) -> dict:
    """Monte Carlo Bayesian A/B test using a Beta-Binomial conjugate model.

    Default prior: Beta(1, 1) (uniform / no historical knowledge).
    Custom priors shift alpha/beta to incorporate historical data.
    """
    alpha_a = successes_a + prior_alpha
    beta_a = total_a - successes_a + prior_beta
    alpha_b = successes_b + prior_alpha
    beta_b = total_b - successes_b + prior_beta

    rng = np.random.default_rng(42)
    samples_a = rng.beta(alpha_a, beta_a, n_sim)
    samples_b = rng.beta(alpha_b, beta_b, n_sim)

    prob_b_better = float(np.mean(samples_b > samples_a))
    lift_samples = samples_b - samples_a
    expected_lift = float(np.mean(lift_samples))
    ci_low = float(np.percentile(lift_samples, 2.5))
    ci_high = float(np.percentile(lift_samples, 97.5))
    expected_loss = float(np.mean(np.maximum(samples_a - samples_b, 0)))

    return {
        "prob_b_better": prob_b_better,
        "expected_lift": expected_lift,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "expected_loss": expected_loss,
        "alpha_a": alpha_a,
        "beta_a": beta_a,
        "alpha_b": alpha_b,
        "beta_b": beta_b,
        "rate_a": successes_a / total_a if total_a > 0 else 0.0,
        "rate_b": successes_b / total_b if total_b > 0 else 0.0,
    }


def bayesian_rpu_test(
    rev_a: float, sends_a: int, orders_a: int,
    rev_b: float, sends_b: int, orders_b: int,
    cv: float = 1.0, n_sim: int = 10_000,
) -> dict:
    """Bayesian test for Revenue Per User using a Normal approximation.

    RPU is continuous and zero-inflated (most users generate £0). We estimate
    the variance via a mixture decomposition:
        Var(RPU) = order_rate × AOV² × (CV² + 1 − order_rate)
    then simulate from Normal posteriors for each group's mean RPU.
    """
    rpu_a = rev_a / sends_a if sends_a > 0 else 0.0
    rpu_b = rev_b / sends_b if sends_b > 0 else 0.0

    aov_a = rev_a / orders_a if orders_a > 0 else 0.0
    aov_b = rev_b / orders_b if orders_b > 0 else 0.0
    p_a = orders_a / sends_a if sends_a > 0 else 0.0
    p_b = orders_b / sends_b if sends_b > 0 else 0.0

    var_a = p_a * aov_a ** 2 * (cv ** 2 + 1 - p_a) if p_a > 0 else 0.01 ** 2
    var_b = p_b * aov_b ** 2 * (cv ** 2 + 1 - p_b) if p_b > 0 else 0.01 ** 2

    se_a = np.sqrt(var_a / sends_a) if sends_a > 0 else 0.01
    se_b = np.sqrt(var_b / sends_b) if sends_b > 0 else 0.01

    rng = np.random.default_rng(43)
    samples_a = rng.normal(rpu_a, se_a, n_sim)
    samples_b = rng.normal(rpu_b, se_b, n_sim)

    prob_b_better = float(np.mean(samples_b > samples_a))
    lift_samples = samples_b - samples_a
    expected_lift = float(np.mean(lift_samples))
    ci_low = float(np.percentile(lift_samples, 2.5))
    ci_high = float(np.percentile(lift_samples, 97.5))
    expected_loss = float(np.mean(np.maximum(samples_a - samples_b, 0)))

    return {
        "prob_b_better": prob_b_better,
        "expected_lift": expected_lift,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "expected_loss": expected_loss,
        "rpu_a": rpu_a,
        "rpu_b": rpu_b,
        "se_a": se_a,
        "se_b": se_b,
    }


# ---------------------------------------------------------------------------
# Frequentist verdict helper
# ---------------------------------------------------------------------------

def verdict_card(
    label: str,
    result: dict,
    is_pct: bool = True,
    alpha: float = 0.05,
) -> None:
    if is_pct:
        def f(v: float) -> str:
            return f"{v:.2%}"
        control_val = f"{result['rate_a']:.2%}"
        variant_val = f"{result['rate_b']:.2%}"
    else:
        def f(v: float) -> str:
            return f"£{v:.2f}"
        control_val = f"£{result['mean_a']:.2f}"
        variant_val = f"£{result['mean_b']:.2f}"

    lift = result["lift"]
    p = result["p_value"]
    lo_a, hi_a = result["ci_a"]
    lo_b, hi_b = result["ci_b"]

    overlaps = lo_a <= hi_b and lo_b <= hi_a
    if overlaps:
        ci_note = "Ranges <b>overlap</b> — can't be sure the difference is real."
        ci_color = "#F57F17"
    else:
        ci_note = "Ranges <b>don't overlap</b> — the difference is real."
        ci_color = "#2E7D32"

    significant = p < alpha
    if significant and lift > 0:
        emoji = "🟢"
        color = "#2E7D32"
        bg = "#E8F5E9"
        status = "Variant wins"
        takeaway = f"Variant is <b>{abs(lift):.2%} higher</b> — this result is real, not due to chance."
    elif significant and lift < 0:
        emoji = "🔴"
        color = "#C62828"
        bg = "#FFEBEE"
        status = "Control wins"
        takeaway = f"Variant is <b>{abs(lift):.2%} lower</b> — Control is performing better here."
    elif significant:
        emoji = "⚪"
        color = "#616161"
        bg = "#F5F5F5"
        status = "No difference"
        takeaway = "Both groups are performing the same."
    else:
        emoji = "🟡"
        color = "#F57F17"
        bg = "#FFF8E1"
        status = "Not enough evidence"
        takeaway = (
            f"There's a <b>{abs(lift):.2%}</b> difference but we <b>can't be confident</b> "
            f"it's real — it could be random variation. Collect more data or re-test."
        )

    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background: {bg};
                    padding: 14px 18px; border-radius: 6px; margin-bottom: 10px;">
            <div style="font-size: 16px; font-weight: 700; color: {color}; margin-bottom: 8px;">
                {emoji} {label}: {status}
            </div>
            <div style="display: flex; gap: 24px; font-size: 13px; margin-bottom: 8px;">
                <span><span style="color:#888;">Control</span> <b>{control_val}</b></span>
                <span><span style="color:#888;">Variant</span> <b>{variant_val}</b></span>
                <span><span style="color:#888;">Lift</span> <b>{lift:+.2%}</b></span>
                <span><span style="color:#888;">p-value</span> <b>{fmt_p(p)}</b></span>
            </div>
            <div style="font-size: 13px; color: #444; margin-bottom: 8px;">{takeaway}</div>
            <div style="font-size: 12px; color: #666; border-top: 1px solid rgba(0,0,0,0.08);
                        padding-top: 8px; line-height: 1.6;">
                <span style="color:#1565C0;">Control</span> {f(lo_a)} – {f(hi_a)}
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <span style="color:#2E7D32;">Variant</span> {f(lo_b)} – {f(hi_b)}
                <br>
                <span style="color:{ci_color}; font-weight: 600;">{ci_note}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Bayesian verdict helper
# ---------------------------------------------------------------------------

def bayesian_verdict_card(metric_label: str, bayes: dict) -> None:
    """Styled card summarising Bayesian A/B test results in plain language."""
    prob = bayes["prob_b_better"]
    prob_str = fmt_prob(prob)
    lift = bayes["expected_lift"]
    ci_lo = bayes["ci_low"]
    ci_hi = bayes["ci_high"]
    loss = bayes["expected_loss"]
    rate_a = bayes["rate_a"]
    rate_b = bayes["rate_b"]

    if prob >= 0.95:
        emoji, color, bg = "🟢", "#2E7D32", "#E8F5E9"
        status = "Variant wins"
        takeaway = (
            f"We're <b>{prob_str} confident</b> that Variant genuinely beats Control. "
            f"The likely improvement is around <b>{abs(lift) * 100:.2f} percentage points</b>. Safe to roll out."
        )
    elif prob >= 0.75:
        emoji, color, bg = "🟡", "#F57F17", "#FFF8E1"
        status = "Variant looks promising"
        takeaway = (
            f"There's a <b>{prob_str} chance</b> that Variant is better "
            f"(by roughly <b>{abs(lift) * 100:.2f} percentage points</b>), "
            f"but we'd recommend gathering more data before committing to a full rollout."
        )
    elif prob > 0.25:
        emoji, color, bg = "⚪", "#616161", "#F5F5F5"
        status = "Too close to call"
        takeaway = (
            f"There's only a <b>{prob_str} chance</b> that Variant is better — "
            f"we can't confidently pick a winner. Keep the test running or re-test with a larger audience."
        )
    elif prob > 0.05:
        emoji, color, bg = "🟡", "#F57F17", "#FFF8E1"
        status = "Control looks stronger"
        takeaway = (
            f"There's only a <b>{prob_str} chance</b> that Variant beats Control. "
            f"Control appears to be performing better here."
        )
    else:
        emoji, color, bg = "🔴", "#C62828", "#FFEBEE"
        status = "Control wins"
        takeaway = (
            f"There's only a <b>{prob_str} chance</b> that Variant beats Control. "
            f"Stick with Control."
        )

    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background: {bg};
                    padding: 14px 18px; border-radius: 6px; margin-bottom: 10px;">
            <div style="font-size: 16px; font-weight: 700; color: {color}; margin-bottom: 8px;">
                {emoji} {metric_label}: {status}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 24px; font-size: 13px; margin-bottom: 8px;">
                <span><span style="color:#888;">Control</span> <b>{rate_a:.2%}</b></span>
                <span><span style="color:#888;">Variant</span> <b>{rate_b:.2%}</b></span>
                <span><span style="color:#888;">Chance Variant wins</span> <b>{prob_str}</b></span>
                <span><span style="color:#888;">Likely improvement</span> <b>{lift * 100:+.2f} percentage points</b></span>
            </div>
            <div style="font-size: 13px; color: #444; margin-bottom: 8px;">{takeaway}</div>
            <div style="font-size: 12px; color: #666; border-top: 1px solid rgba(0,0,0,0.08);
                        padding-top: 8px; line-height: 1.6;">
                <b>Realistic range for the difference:</b> {ci_lo * 100:+.2f} to {ci_hi * 100:+.2f} percentage points
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <b>Risk if Variant is actually worse:</b> {loss * 100:.2f} percentage points
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bayesian_rpu_verdict_card(bayes: dict) -> None:
    """Styled card for Revenue Per User Bayesian results (£-formatted)."""
    prob = bayes["prob_b_better"]
    prob_str = fmt_prob(prob)
    lift = bayes["expected_lift"]
    ci_lo = bayes["ci_low"]
    ci_hi = bayes["ci_high"]
    loss = bayes["expected_loss"]
    rpu_a = bayes["rpu_a"]
    rpu_b = bayes["rpu_b"]

    if prob >= 0.95:
        emoji, color, bg = "🟢", "#2E7D32", "#E8F5E9"
        status = "Variant wins"
        takeaway = (
            f"We're <b>{prob_str} confident</b> that Variant generates more revenue per user. "
            f"The likely improvement is around <b>£{abs(lift):.2f} per user</b>. Safe to roll out."
        )
    elif prob >= 0.75:
        emoji, color, bg = "🟡", "#F57F17", "#FFF8E1"
        status = "Variant looks promising"
        takeaway = (
            f"There's a <b>{prob_str} chance</b> that Variant generates more per user "
            f"(by roughly <b>£{abs(lift):.2f}</b>), "
            f"but we'd recommend gathering more data before committing."
        )
    elif prob > 0.25:
        emoji, color, bg = "⚪", "#616161", "#F5F5F5"
        status = "Too close to call"
        takeaway = (
            f"There's only a <b>{prob_str} chance</b> that Variant generates more per user — "
            f"we can't confidently pick a winner. Keep testing."
        )
    elif prob > 0.05:
        emoji, color, bg = "🟡", "#F57F17", "#FFF8E1"
        status = "Control looks stronger"
        takeaway = (
            f"There's only a <b>{prob_str} chance</b> that Variant beats Control on RPU. "
            f"Control appears to generate more revenue per user."
        )
    else:
        emoji, color, bg = "🔴", "#C62828", "#FFEBEE"
        status = "Control wins"
        takeaway = (
            f"There's only a <b>{prob_str} chance</b> that Variant beats Control on RPU. "
            f"Stick with Control."
        )

    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background: {bg};
                    padding: 14px 18px; border-radius: 6px; margin-bottom: 10px;">
            <div style="font-size: 16px; font-weight: 700; color: {color}; margin-bottom: 8px;">
                {emoji} Revenue Per User: {status}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 24px; font-size: 13px; margin-bottom: 8px;">
                <span><span style="color:#888;">Control</span> <b>£{rpu_a:.2f}</b></span>
                <span><span style="color:#888;">Variant</span> <b>£{rpu_b:.2f}</b></span>
                <span><span style="color:#888;">Chance Variant wins</span> <b>{prob_str}</b></span>
                <span><span style="color:#888;">Likely improvement</span> <b>£{lift:+.2f} per user</b></span>
            </div>
            <div style="font-size: 13px; color: #444; margin-bottom: 8px;">{takeaway}</div>
            <div style="font-size: 12px; color: #666; border-top: 1px solid rgba(0,0,0,0.08);
                        padding-top: 8px; line-height: 1.6;">
                <b>Realistic range for the difference:</b> £{ci_lo:+.2f} to £{ci_hi:+.2f} per user
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <b>Risk if Variant is actually worse:</b> £{loss:.2f} per user
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# App layout
# ===========================================================================

st.set_page_config(page_title="CRM Experiment Lab", layout="wide")
st.title("CRM Experiment Lab")
st.caption("In-house tool for CRM experimentation.")

with st.expander("Which tab should I use — Frequentist or Bayesian?"):
    st.markdown("""
| When | Use |
|---|---|
| **Early or mid-test** — you want a read before the test is done | **Bayesian** — gives you a probability and revenue scenarios even with limited data. |
| **Final reporting** — test is done, you need a clear verdict for the tracker or stakeholders | **Frequentist** — gives you the standard "significant" or "not significant" result. |
| **You want both** — the official verdict plus revenue upside/risk | **Both tabs** |

**Bayesian** — "How likely is Variant better, and what could it mean for revenue?"
- Probability (e.g. "87% chance Variant is better") and worst / most likely / best case revenue.
- Useful with smaller sends; you still get a signal instead of "not enough data".
- Revenue Per User picks up cases where Variant has fewer orders but higher revenue.

**Frequentist** — "Did it work?"
- Clear verdict: Variant wins, Control wins, or not enough evidence.
- Works best with decent send size (e.g. 10K+ per group). What most people expect for "statistical significance".

**TL;DR:** Use **Bayesian** to monitor during the test; use **Frequentist** to conclude and report.
""")

# -------------------------------------------------------------------
# Shared inputs (used by both Frequentist and Bayesian tabs)
# -------------------------------------------------------------------

st.subheader("Experiment Context")
experiment_description = st.text_area(
    "What is this experiment testing?",
    placeholder="e.g. Testing a new subject line with urgency messaging vs our standard BAU subject line for the weekly promo email.",
    height=80,
    key="experiment_desc",
)

st.subheader("Campaign Data")
col_ctrl, col_var = st.columns(2)

with col_ctrl:
    st.markdown("#### Control")
    sample_a = int_input("Total Sends", 50000, key="sample_a")
    opens_a = int_input("Unique Opens", 12000, key="opens_a")
    clicks_a = int_input("Unique Clicks", 3500, key="clicks_a")
    orders_a = int_input("Orders", 500, key="orders_a")
    revenue_a = float_input("Total Revenue (£)", 25000.0, key="revenue_a")

with col_var:
    st.markdown("#### Variant")
    sample_b = int_input("Total Sends", 50000, key="sample_b")
    opens_b = int_input("Unique Opens", 13500, key="opens_b")
    clicks_b = int_input("Unique Clicks", 4200, key="clicks_b")
    orders_b = int_input("Orders", 620, key="orders_b")
    revenue_b = float_input("Total Revenue (£)", 31000.0, key="revenue_b")

all_values = [
    sample_a, opens_a, clicks_a, orders_a, revenue_a,
    sample_b, opens_b, clicks_b, orders_b, revenue_b,
]
if any(v is None for v in all_values):
    st.warning("Please fix the highlighted inputs above.")
    st.stop()

errors: list[str] = []
for label, count, total in [
    ("Control Unique Opens", opens_a, sample_a),
    ("Control Unique Clicks", clicks_a, sample_a),
    ("Control Orders", orders_a, sample_a),
    ("Variant Unique Opens", opens_b, sample_b),
    ("Variant Unique Clicks", clicks_b, sample_b),
    ("Variant Orders", orders_b, sample_b),
]:
    if count > total:
        errors.append(f"{label} ({count:,}) cannot exceed Total Sends ({total:,}).")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

if sample_a == 0 or sample_b == 0:
    st.warning("Total Sends must be greater than 0 for both groups.")
    st.stop()

st.divider()

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tab_freq, tab_bayes = st.tabs(["Frequentist", "Bayesian"])

# ===================================================================
# Frequentist tab
# ===================================================================

with tab_freq:

    with st.expander("How it works"):
        st.markdown("""
**What is this?**
A tool that tells you whether your email A/B test produced a real winner or if the difference was just noise.

**How to use it**
1. Describe what you're testing (e.g. "new subject line vs BAU").
2. Enter your Control and Variant numbers — sends, opens, clicks, orders and revenue.
3. Hit enter and the app does the rest.

**What you'll get**
| Step | What it tells you |
|---|---|
| **Step 1 — Did the Variant win?** | A clear verdict for each metric (Open Rate, CTR, Conversion Rate, AOV) plus the confidence range so you can see how certain we are. |
| **Step 2 — Revenue impact** | If we rolled out to the full audience, how much more (or less) revenue would the Variant generate? |
| **Step 3 — Share findings** | An AI-written summary you can paste into Slack or email, and a one-click copy row for your Google Sheets tracker. |
| **Step 4 — Detailed stats** | The raw numbers (z-scores, t-scores, p-values) for anyone who wants them. |

**Reading the results**
- 🟢 **Variant wins** — the difference is real. Safe to roll out.
- 🔴 **Control wins** — the difference is real, but Control is better. Stick with it.
- 🟡 **Not enough evidence** — we can't tell yet. Collect more data or re-test.

**What counts as "real"?**
The app checks whether there's less than a 5% chance the difference happened by luck. If so, we call it a real difference. The confidence ranges underneath each verdict show you the ballpark the true value sits in — if the Control and Variant ranges don't overlap, you can be confident.
""")

    # -------------------------------------------------------------------
    # Run tests
    # -------------------------------------------------------------------

    open_result = z_test_proportions(opens_a, sample_a, opens_b, sample_b)
    ctr_result = z_test_proportions(clicks_a, sample_a, clicks_b, sample_b)
    cvr_result = z_test_proportions(orders_a, sample_a, orders_b, sample_b)
    aov_result = t_test_aov(revenue_a, orders_a, revenue_b, orders_b)

    # -------------------------------------------------------------------
    # Step 1: Results — Did the Variant Win?
    # -------------------------------------------------------------------

    st.subheader("Step 1 — Did the Variant Win?")
    st.caption(
        "🟢 = Variant wins · 🔴 = Control wins · 🟡 = Not enough evidence (need more data) "
        "· 95% confidence ranges shown below each verdict"
    )

    v_left, v_right = st.columns(2)
    with v_left:
        verdict_card("Open Rate", open_result)
        verdict_card("Conversion Rate (Orders)", cvr_result)
    with v_right:
        verdict_card("Click-Through Rate", ctr_result)
        verdict_card("AOV (Avg Order Value)", aov_result, is_pct=False)

    # -------------------------------------------------------------------
    # Step 2: Revenue impact
    # -------------------------------------------------------------------

    st.subheader("Step 2 — What's the Revenue Impact?")

    with st.expander("Rollout Revenue Breakdown"):
        total_audience = sample_a + sample_b
        cvr_control = cvr_result["rate_a"]
        cvr_variant = cvr_result["rate_b"]
        aov_variant = aov_result["mean_b"]

        rev_if_control = total_audience * cvr_control * aov_variant
        rev_if_variant = total_audience * cvr_variant * aov_variant
        revenue_uplift = rev_if_variant - rev_if_control

        orders_if_control = total_audience * cvr_control
        orders_if_variant = total_audience * cvr_variant

        uplift_color = "#2E7D32" if revenue_uplift >= 0 else "#C62828"

        st.markdown(
            f"""
<table style="width:100%; border-collapse:collapse; font-size:14px; margin-bottom:12px;">
  <thead>
    <tr style="border-bottom:2px solid #ddd;">
      <th style="text-align:left; padding:8px 12px;"></th>
      <th style="text-align:right; padding:8px 12px; color:#1565C0;">Control rollout</th>
      <th style="text-align:right; padding:8px 12px; color:#2E7D32;">Variant rollout</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:6px 12px;">Audience</td>
      <td style="text-align:right; padding:6px 12px;">{total_audience:,}</td>
      <td style="text-align:right; padding:6px 12px;">{total_audience:,}</td>
    </tr>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:6px 12px;">Conversion Rate</td>
      <td style="text-align:right; padding:6px 12px;">{cvr_control:.2%}</td>
      <td style="text-align:right; padding:6px 12px;">{cvr_variant:.2%}</td>
    </tr>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:6px 12px;">Estimated Orders</td>
      <td style="text-align:right; padding:6px 12px;">{orders_if_control:,.0f}</td>
      <td style="text-align:right; padding:6px 12px;">{orders_if_variant:,.0f}</td>
    </tr>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:6px 12px;">AOV (Variant)</td>
      <td style="text-align:right; padding:6px 12px;">£{aov_variant:.2f}</td>
      <td style="text-align:right; padding:6px 12px;">£{aov_variant:.2f}</td>
    </tr>
    <tr style="border-top:2px solid #ddd; font-weight:700;">
      <td style="padding:8px 12px;">Estimated Revenue</td>
      <td style="text-align:right; padding:8px 12px; color:#1565C0;">£{rev_if_control:,.2f}</td>
      <td style="text-align:right; padding:8px 12px; color:#2E7D32;">£{rev_if_variant:,.2f}</td>
    </tr>
  </tbody>
</table>
<div style="font-size:15px; font-weight:700; color:{uplift_color}; margin-top:4px;">
  Revenue uplift from Variant: £{revenue_uplift:+,.2f}
  <span style="font-weight:400; color:#666; font-size:13px;">
    &nbsp;(£{rev_if_variant:,.2f} − £{rev_if_control:,.2f})
  </span>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------
    # Step 3: Share findings
    # -------------------------------------------------------------------

    st.subheader("Step 3 — Share Findings")

    with st.expander("Summarise Findings"):
        st.caption("Summary is generated via an AI model based on your experiment context and results.")
        summary_context = (
            f"Sample sizes: Control {sample_a:,} sends, Variant {sample_b:,} sends.\n\n"
            f"Open Rate: Control {open_result['rate_a']:.2%} vs Variant {open_result['rate_b']:.2%}, "
            f"lift {open_result['lift']:+.2%}, p-value {fmt_p(open_result['p_value'])}, "
            f"{'significant' if open_result['p_value'] < 0.05 else 'not significant'}.\n"
            f"CTR: Control {ctr_result['rate_a']:.2%} vs Variant {ctr_result['rate_b']:.2%}, "
            f"lift {ctr_result['lift']:+.2%}, p-value {fmt_p(ctr_result['p_value'])}, "
            f"{'significant' if ctr_result['p_value'] < 0.05 else 'not significant'}.\n"
            f"Conversion Rate: Control {cvr_result['rate_a']:.2%} vs Variant {cvr_result['rate_b']:.2%}, "
            f"lift {cvr_result['lift']:+.2%}, p-value {fmt_p(cvr_result['p_value'])}, "
            f"{'significant' if cvr_result['p_value'] < 0.05 else 'not significant'}.\n"
            f"AOV: Control £{aov_result['mean_a']:.2f} vs Variant £{aov_result['mean_b']:.2f}, "
            f"lift {aov_result['lift']:+.2%}, p-value {fmt_p(aov_result['p_value'])}, "
            f"{'significant' if aov_result['p_value'] < 0.05 else 'not significant'}.\n"
        )

        if st.button("Generate Summary"):
            import requests as req

            if not OPENROUTER_API_KEY:
                st.error(
                    "OpenRouter API key not set. Add OPENROUTER_API_KEY to your environment, "
                    "or create a .env file with OPENROUTER_API_KEY=your_key. See .env.example."
                )
            else:
                with st.spinner("Generating summary..."):
                    try:
                        resp = req.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                "Content-Type": "application/json",
                            },
                        json={
                            "model": "openrouter/free",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a CRM experimentation analyst. Your summary will be copy-pasted directly "
                                        "into Slack or email by the CRM team — so write in full sentences, include all key numbers, "
                                        "and make it ready to send with no extra editing.\n\n"
                                        "Structure the output EXACTLY like this:\n\n"
                                        "**CRM Experiment Summary**\n\n"
                                        "**What we tested:** [1–2 sentences using the experiment context — e.g. subject line test, segment, send date]\n\n"
                                        "**Sample:** Control [X] sends, Variant [Y] sends.\n\n"
                                        "**Key results:**\n"
                                        "- **Open Rate:** Control [X%] vs Variant [Y%] — [lift in pp]. [One short sentence: real difference or could be random.]\n"
                                        "- **Click-through rate (CTR):** Control [X%] vs Variant [Y%] — [lift]. [Real difference or could be random.]\n"
                                        "- **Conversion rate:** Control [X%] vs Variant [Y%] — [lift]. [Real difference or could be random.]\n"
                                        "- **AOV:** Control £[X] vs Variant £[Y] — [lift]. [Real difference or could be random.]\n\n"
                                        "**Winner:** [Variant / Control / No clear winner]. [One sentence: what we’re rolling out or what we’re doing next.]\n\n"
                                        "**Bottom line:** [One sentence the CRM lead can use in a standup or forward to stakeholders.]\n\n"
                                        "Rules:\n"
                                        "- Use the exact numbers from the results provided — do not round or approximate.\n"
                                        "- Use plain language: 'real difference' not 'statistically significant', 'could be random' not 'not significant'.\n"
                                        "- Write so that someone can copy the whole block into Slack or email and send it as-is.\n"
                                        "- Use GBP (£) for currency. Keep each bullet to 1–2 sentences.\n"
                                        "- Do NOT add a word count, disclaimer, or extra recommendation section.\n"
                                        "- Do NOT use phrases like 'Based on the data provided' — write as if reporting the experiment."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"Experiment context: {experiment_description or 'Not provided'}\n\n"
                                        f"Write a test summary for these results:\n\n{summary_context}"
                                    ),
                                },
                            ],
                        },
                        timeout=30,
                    )
                        resp.raise_for_status()
                        summary = resp.json()["choices"][0]["message"]["content"]
                        st.markdown("---")
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Could not generate summary: {e}")

    st.markdown("#### Copy to Google Sheets")
    st.caption("Select all rows below and paste straight into your results tracker.")

    def sig_label(p: float) -> str:
        return "Yes" if p < 0.05 else "No"

    total_audience = sample_a + sample_b
    cvr_ctrl = cvr_result["rate_a"]
    cvr_var = cvr_result["rate_b"]
    aov_var = aov_result["mean_b"]
    rev_ctrl_rollout = total_audience * cvr_ctrl * aov_var
    rev_var_rollout = total_audience * cvr_var * aov_var

    gsheet_df = pd.DataFrame([{
        "Control Sample": sample_a,
        "Control Open Rate": f"{open_result['rate_a']:.2%}",
        "Control CTR": f"{ctr_result['rate_a']:.2%}",
        "Control CVR": f"{cvr_result['rate_a']:.2%}",
        "Control AOV": f"£{aov_result['mean_a']:.2f}",
        "Control Revenue": f"£{revenue_a:,.2f}",
        "Variant Sample": sample_b,
        "Variant Open Rate": f"{open_result['rate_b']:.2%}",
        "Variant CTR": f"{ctr_result['rate_b']:.2%}",
        "Variant CVR": f"{cvr_result['rate_b']:.2%}",
        "Variant AOV": f"£{aov_result['mean_b']:.2f}",
        "Variant Revenue": f"£{revenue_b:,.2f}",
        "Open Rate Lift": f"{open_result['lift']:+.2%}",
        "CTR Lift": f"{ctr_result['lift']:+.2%}",
        "CVR Lift": f"{cvr_result['lift']:+.2%}",
        "AOV Lift": f"{aov_result['lift']:+.2%}",
        "Open Rate p-value": fmt_p(open_result["p_value"]),
        "CTR p-value": fmt_p(ctr_result["p_value"]),
        "CVR p-value": fmt_p(cvr_result["p_value"]),
        "AOV p-value": fmt_p(aov_result["p_value"]),
        "Open Rate Sig?": sig_label(open_result["p_value"]),
        "CTR Sig?": sig_label(ctr_result["p_value"]),
        "CVR Sig?": sig_label(cvr_result["p_value"]),
        "AOV Sig?": sig_label(aov_result["p_value"]),
        "Rollout Rev (Control)": f"£{rev_ctrl_rollout:,.2f}",
        "Rollout Rev (Variant)": f"£{rev_var_rollout:,.2f}",
        "Revenue Uplift": f"£{rev_var_rollout - rev_ctrl_rollout:+,.2f}",
    }])

    tsv_string = gsheet_df.to_csv(sep="\t", index=False, header=False)
    st.code(tsv_string.strip(), language=None)
    st.caption("Click the copy icon in the top-right of the box above, then paste into Google Sheets.")

    # -------------------------------------------------------------------
    # Step 4: Detailed stats (for advanced users)
    # -------------------------------------------------------------------

    with st.expander("Step 4 — Detailed Test Results (Advanced)"):
        st.dataframe(
            [
                {
                    "Metric": "Open Rate",
                    "Control": f"{open_result['rate_a']:.2%}",
                    "Variant": f"{open_result['rate_b']:.2%}",
                    "Lift": f"{open_result['lift']:+.2%}",
                    "Z-score": f"{open_result['z']:.3f}",
                    "p-value": fmt_p(open_result["p_value"]),
                },
                {
                    "Metric": "CTR",
                    "Control": f"{ctr_result['rate_a']:.2%}",
                    "Variant": f"{ctr_result['rate_b']:.2%}",
                    "Lift": f"{ctr_result['lift']:+.2%}",
                    "Z-score": f"{ctr_result['z']:.3f}",
                    "p-value": fmt_p(ctr_result["p_value"]),
                },
                {
                    "Metric": "Conversion Rate",
                    "Control": f"{cvr_result['rate_a']:.2%}",
                    "Variant": f"{cvr_result['rate_b']:.2%}",
                    "Lift": f"{cvr_result['lift']:+.2%}",
                    "Z-score": f"{cvr_result['z']:.3f}",
                    "p-value": fmt_p(cvr_result["p_value"]),
                },
                {
                    "Metric": "AOV",
                    "Control": f"£{aov_result['mean_a']:.2f}",
                    "Variant": f"£{aov_result['mean_b']:.2f}",
                    "Lift": f"{aov_result['lift']:+.2%}",
                    "Z-score": f"{aov_result['t']:.3f} (t)",
                    "p-value": fmt_p(aov_result["p_value"]),
                },
            ],
            use_container_width=True,
            hide_index=True,
        )

# ===================================================================
# Bayesian tab
# ===================================================================

with tab_bayes:

    with st.expander("How it works — Bayesian tab explained"):
        st.markdown("""
**What does this tab do?**

The Frequentist tab tells you *"is the difference real or random noise?"* — a yes/no answer.

This Bayesian tab goes further and answers a more natural question: **"What is the *chance* that Variant is actually better than Control?"** — giving you a percentage you can act on.

**How does it work?**

Behind the scenes, the tool simulates your test 10,000 times, each time estimating what the true rate might be for both groups. It then counts how often Variant comes out ahead. No jargon, just: *"Out of 10,000 simulations, Variant won X% of the time."*

**What you'll see**

| What you'll see | What it means |
|---|---|
| **Chance Variant wins** | A direct percentage — e.g. "92% chance Variant is better". Much easier to act on than a p-value. |
| **Likely improvement** | How much better (or worse) the Variant is, in percentage points (for rates) or in £ (for revenue). |
| **Realistic range** | The best-case and worst-case improvement. Think of it as "the real answer almost certainly falls somewhere in this range". |
| **Risk if wrong** | If you roll out the Variant and it turns out to be worse, this is how much you'd lose on average. A tiny number here means low risk. |

**Metrics analysed**

| Metric | What it measures | Why it matters |
|---|---|---|
| **Open Rate** | % of recipients who opened the email | Did the subject line / sender name work? |
| **CTR** | % of recipients who clicked | Did the email content drive action? |
| **Order Rate** | % of recipients who placed an order | Did the email convert? |
| **Revenue Per User (RPU)** | Average £ revenue generated per recipient | The bottom line — captures *both* conversion rate and order value in a single number. A variant can have fewer orders but higher RPU if order values are larger. |

**How to read the verdict**
- 🟢 **≥95% chance** — Variant wins. Confident enough to roll out.
- 🟡 **75–95% chance** — Looking good, but consider more data before going all-in.
- ⚪ **25–75% chance** — Too close to call. Keep testing.
- 🔴 **<25% chance** — Control is likely better. Stick with what you have.

**What are priors?**

If you've run similar campaigns before, you can tell the tool "historically, our open rate is around 25%". This makes the analysis more robust, especially with small sample sizes. Toggle on "Include historical priors" below to use this. If you're unsure, leave it off — the tool works perfectly fine without it.

**Why use this instead of (or alongside) the Frequentist tab?**

The Frequentist tab gives you a binary "significant or not" — useful but blunt. This tab tells you *how confident* you should be and *what the realistic upside and downside look like*, which is often more useful when deciding whether to roll out an email change or keep testing.
""")

    # -------------------------------------------------------------------
    # Prior toggle
    # -------------------------------------------------------------------

    use_priors = st.toggle(
        "Include historical priors",
        value=False,
        key="use_priors",
        help="Factor in historical campaign performance. Makes the analysis more robust with smaller samples.",
    )

    if use_priors:
        st.caption(
            "Enter your typical historical rates and the average sends per campaign. "
            "This anchors the analysis so small tests don't swing wildly — without "
            "overwhelming the current test data."
        )
        prior_n = st.number_input(
            "Avg sends per campaign", min_value=100, max_value=10_000_000,
            value=50000, step=5000, key="prior_n",
        )
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            prior_open_rate = st.number_input(
                "Historical Open Rate (%)", min_value=0.1, max_value=99.9,
                value=25.0, step=0.5, key="prior_open_rate",
            )
        with pc2:
            prior_ctr_rate = st.number_input(
                "Historical CTR (%)", min_value=0.1, max_value=99.9,
                value=7.0, step=0.5, key="prior_ctr_rate",
            )
        with pc3:
            prior_order_rate = st.number_input(
                "Historical Order Rate (%)", min_value=0.01, max_value=99.9,
                value=1.0, step=0.1, key="prior_order_rate",
            )

        open_pa = (prior_open_rate / 100) * prior_n
        open_pb = (1 - prior_open_rate / 100) * prior_n
        ctr_pa = (prior_ctr_rate / 100) * prior_n
        ctr_pb = (1 - prior_ctr_rate / 100) * prior_n
        order_pa = (prior_order_rate / 100) * prior_n
        order_pb = (1 - prior_order_rate / 100) * prior_n
    else:
        open_pa, open_pb = 1.0, 1.0
        ctr_pa, ctr_pb = 1.0, 1.0
        order_pa, order_pb = 1.0, 1.0

    # -------------------------------------------------------------------
    # Run Bayesian tests for all metrics
    # -------------------------------------------------------------------

    bayes_open = bayesian_ab_test(
        opens_a, sample_a, opens_b, sample_b,
        prior_alpha=open_pa, prior_beta=open_pb,
    )
    bayes_ctr = bayesian_ab_test(
        clicks_a, sample_a, clicks_b, sample_b,
        prior_alpha=ctr_pa, prior_beta=ctr_pb,
    )
    bayes_order = bayesian_ab_test(
        orders_a, sample_a, orders_b, sample_b,
        prior_alpha=order_pa, prior_beta=order_pb,
    )
    bayes_rpu = bayesian_rpu_test(
        revenue_a, sample_a, orders_a,
        revenue_b, sample_b, orders_b,
    )

    # -------------------------------------------------------------------
    # Step 1: Verdict cards (2×2 layout)
    # -------------------------------------------------------------------

    st.subheader("Step 1 — What's the Chance Variant is Better?")
    st.caption(
        "🟢 = Confident it's better (≥95%) · 🟡 = Looks promising (75–95%) "
        "· ⚪ = Too close to call · 🔴 = Control is likely better"
    )

    bv_left, bv_right = st.columns(2)
    with bv_left:
        bayesian_verdict_card("Open Rate", bayes_open)
    with bv_right:
        bayesian_verdict_card("Click-Through Rate", bayes_ctr)

    bv_left2, bv_right2 = st.columns(2)
    with bv_left2:
        bayesian_verdict_card("Order Rate", bayes_order)
    with bv_right2:
        bayesian_rpu_verdict_card(bayes_rpu)

    # -------------------------------------------------------------------
    # Step 2: Distribution charts (2×2 grid)
    # -------------------------------------------------------------------

    st.subheader("Step 2 — Where Does the True Value Sit?")
    st.caption(
        "Each chart shows the range of likely true values for Control (blue) and Variant (green). "
        "The more the curves are separated, the more confident we can be there's a real difference."
    )

    def _make_beta_chart(b: dict, label: str) -> go.Figure:
        a_a, b_a = b["alpha_a"], b["beta_a"]
        a_b, b_b = b["alpha_b"], b["beta_b"]
        mu_a = a_a / (a_a + b_a)
        mu_b = a_b / (a_b + b_b)
        s_a = np.sqrt(a_a * b_a / ((a_a + b_a) ** 2 * (a_a + b_a + 1)))
        s_b = np.sqrt(a_b * b_b / ((a_b + b_b) ** 2 * (a_b + b_b + 1)))
        spread = max(s_a, s_b)
        lo = max(0, min(mu_a, mu_b) - 5 * spread)
        hi = min(1, max(mu_a, mu_b) + 5 * spread)
        x = np.linspace(lo, hi, 500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, a_a, b_a), mode="lines", name="Control",
            line=dict(color="#1565C0", width=2.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.12)",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, a_b, b_b), mode="lines", name="Variant",
            line=dict(color="#2E7D32", width=2.5),
            fill="tozeroy", fillcolor="rgba(46,125,50,0.12)",
        ))
        fig.update_layout(
            title=dict(text=label, font=dict(size=14)),
            xaxis_title="Likely true rate",
            yaxis_title="",
            template="plotly_white",
            height=320,
            margin=dict(t=40, b=40, l=50, r=20),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            showlegend=True,
        )
        fig.update_xaxes(tickformat=".2%")
        fig.update_yaxes(showticklabels=False)
        return fig

    def _make_rpu_chart(b: dict) -> go.Figure:
        rpu_a, se_a = b["rpu_a"], b["se_a"]
        rpu_b, se_b = b["rpu_b"], b["se_b"]
        spread = max(se_a, se_b)
        lo = max(0, min(rpu_a, rpu_b) - 5 * spread)
        hi = max(rpu_a, rpu_b) + 5 * spread
        x = np.linspace(lo, hi, 500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=stats.norm.pdf(x, rpu_a, se_a), mode="lines", name="Control",
            line=dict(color="#1565C0", width=2.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.12)",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats.norm.pdf(x, rpu_b, se_b), mode="lines", name="Variant",
            line=dict(color="#2E7D32", width=2.5),
            fill="tozeroy", fillcolor="rgba(46,125,50,0.12)",
        ))
        fig.update_layout(
            title=dict(text="Revenue Per User", font=dict(size=14)),
            xaxis_title="Likely true RPU (£)",
            yaxis_title="",
            template="plotly_white",
            height=320,
            margin=dict(t=40, b=40, l=50, r=20),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            showlegend=True,
        )
        fig.update_xaxes(tickprefix="£", tickformat=".2f")
        fig.update_yaxes(showticklabels=False)
        return fig

    ch_r1c1, ch_r1c2 = st.columns(2)
    with ch_r1c1:
        st.plotly_chart(_make_beta_chart(bayes_open, "Open Rate"), use_container_width=True)
    with ch_r1c2:
        st.plotly_chart(_make_beta_chart(bayes_ctr, "Click-Through Rate"), use_container_width=True)

    ch_r2c1, ch_r2c2 = st.columns(2)
    with ch_r2c1:
        st.plotly_chart(_make_beta_chart(bayes_order, "Order Rate"), use_container_width=True)
    with ch_r2c2:
        st.plotly_chart(_make_rpu_chart(bayes_rpu), use_container_width=True)

    # -------------------------------------------------------------------
    # Step 3: Detailed simulation summary
    # -------------------------------------------------------------------

    with st.expander("Step 3 — Full Numbers at a Glance"):
        st.caption(
            "All four metrics side by side. Rate metrics show improvement in percentage points; "
            "RPU shows improvement in £ per user."
        )
        st.dataframe(
            [
                {
                    "Metric": "Open Rate",
                    "Control": f"{bayes_open['rate_a']:.2%}",
                    "Variant": f"{bayes_open['rate_b']:.2%}",
                    "Chance Variant wins": fmt_prob(bayes_open["prob_b_better"]),
                    "Likely improvement": f"{bayes_open['expected_lift'] * 100:+.2f} pts",
                    "Realistic range": f"{bayes_open['ci_low'] * 100:+.2f} to {bayes_open['ci_high'] * 100:+.2f} pts",
                    "Risk if wrong": f"{bayes_open['expected_loss'] * 100:.2f} pts",
                },
                {
                    "Metric": "CTR",
                    "Control": f"{bayes_ctr['rate_a']:.2%}",
                    "Variant": f"{bayes_ctr['rate_b']:.2%}",
                    "Chance Variant wins": fmt_prob(bayes_ctr["prob_b_better"]),
                    "Likely improvement": f"{bayes_ctr['expected_lift'] * 100:+.2f} pts",
                    "Realistic range": f"{bayes_ctr['ci_low'] * 100:+.2f} to {bayes_ctr['ci_high'] * 100:+.2f} pts",
                    "Risk if wrong": f"{bayes_ctr['expected_loss'] * 100:.2f} pts",
                },
                {
                    "Metric": "Order Rate",
                    "Control": f"{bayes_order['rate_a']:.2%}",
                    "Variant": f"{bayes_order['rate_b']:.2%}",
                    "Chance Variant wins": fmt_prob(bayes_order["prob_b_better"]),
                    "Likely improvement": f"{bayes_order['expected_lift'] * 100:+.2f} pts",
                    "Realistic range": f"{bayes_order['ci_low'] * 100:+.2f} to {bayes_order['ci_high'] * 100:+.2f} pts",
                    "Risk if wrong": f"{bayes_order['expected_loss'] * 100:.2f} pts",
                },
                {
                    "Metric": "Revenue Per User",
                    "Control": f"£{bayes_rpu['rpu_a']:.2f}",
                    "Variant": f"£{bayes_rpu['rpu_b']:.2f}",
                    "Chance Variant wins": fmt_prob(bayes_rpu["prob_b_better"]),
                    "Likely improvement": f"£{bayes_rpu['expected_lift']:+.2f}",
                    "Realistic range": f"£{bayes_rpu['ci_low']:+.2f} to £{bayes_rpu['ci_high']:+.2f}",
                    "Risk if wrong": f"£{bayes_rpu['expected_loss']:.2f}",
                },
            ],
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------------------------------------------------
    # Step 4: Bayesian Rollout Impact (RPU-based — the most complete measure)
    # -------------------------------------------------------------------

    st.subheader("Step 4 — If We Rolled Out the Variant, What Happens to Revenue?")
    st.caption(
        "Estimated using Revenue Per User — the single metric that captures both conversion rate "
        "and order value together. This is especially important when one variant has fewer orders "
        "but higher revenue."
    )

    total_sends = sample_a + sample_b

    rpu_rev_pessimistic = total_sends * bayes_rpu["ci_low"]
    rpu_rev_expected = total_sends * bayes_rpu["expected_lift"]
    rpu_rev_optimistic = total_sends * bayes_rpu["ci_high"]

    def _rev_color(v: float) -> str:
        return "#2E7D32" if v >= 0 else "#C62828"

    st.markdown(
        f"""
<table style="width:100%; border-collapse:collapse; font-size:14px; margin-bottom:12px;">
  <thead>
    <tr style="border-bottom:2px solid #ddd;">
      <th style="text-align:left; padding:8px 12px;">Scenario</th>
      <th style="text-align:right; padding:8px 12px;">RPU Change</th>
      <th style="text-align:right; padding:8px 12px;">Revenue Impact</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:6px 12px;">Worst case</td>
      <td style="text-align:right; padding:6px 12px;">£{bayes_rpu['ci_low']:+.2f} per user</td>
      <td style="text-align:right; padding:6px 12px; color:{_rev_color(rpu_rev_pessimistic)}; font-weight:600;">
        £{rpu_rev_pessimistic:+,.2f}
      </td>
    </tr>
    <tr style="border-bottom:1px solid #eee; background:#f9f9f9;">
      <td style="padding:6px 12px; font-weight:700;">Most likely</td>
      <td style="text-align:right; padding:6px 12px; font-weight:700;">£{bayes_rpu['expected_lift']:+.2f} per user</td>
      <td style="text-align:right; padding:6px 12px; color:{_rev_color(rpu_rev_expected)}; font-weight:700;">
        £{rpu_rev_expected:+,.2f}
      </td>
    </tr>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:6px 12px;">Best case</td>
      <td style="text-align:right; padding:6px 12px;">£{bayes_rpu['ci_high']:+.2f} per user</td>
      <td style="text-align:right; padding:6px 12px; color:{_rev_color(rpu_rev_optimistic)}; font-weight:600;">
        £{rpu_rev_optimistic:+,.2f}
      </td>
    </tr>
  </tbody>
</table>
<div style="font-size:12px; color:#888; margin-top:4px;">
  Revenue Impact = {total_sends:,} total sends × RPU change per user
</div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Step 5: Share Bayesian findings
    # -------------------------------------------------------------------

    st.subheader("Step 5 — Share Findings")

    with st.expander("Summarise Findings"):
        st.caption("Summary is generated via an AI model based on your experiment context and results.")
        bayes_summary_context = (
            f"Open Rate: Control {bayes_open['rate_a']:.2%} vs Variant {bayes_open['rate_b']:.2%}, "
            f"{fmt_prob(bayes_open['prob_b_better'])} chance Variant is better, "
            f"expected lift {bayes_open['expected_lift'] * 100:+.2f} percentage points.\n"
            f"CTR: Control {bayes_ctr['rate_a']:.2%} vs Variant {bayes_ctr['rate_b']:.2%}, "
            f"{fmt_prob(bayes_ctr['prob_b_better'])} chance Variant is better, "
            f"expected lift {bayes_ctr['expected_lift'] * 100:+.2f} percentage points.\n"
            f"Order Rate: Control {bayes_order['rate_a']:.2%} vs Variant {bayes_order['rate_b']:.2%}, "
            f"{fmt_prob(bayes_order['prob_b_better'])} chance Variant is better, "
            f"expected lift {bayes_order['expected_lift'] * 100:+.2f} percentage points.\n"
            f"Revenue Per User: Control £{bayes_rpu['rpu_a']:.2f} vs Variant £{bayes_rpu['rpu_b']:.2f}, "
            f"{fmt_prob(bayes_rpu['prob_b_better'])} chance Variant is better, "
            f"expected lift £{bayes_rpu['expected_lift']:+.2f} per user.\n"
            f"Revenue impact if rolled out to {total_sends:,} users: "
            f"worst case £{rpu_rev_pessimistic:+,.2f}, "
            f"most likely £{rpu_rev_expected:+,.2f}, "
            f"best case £{rpu_rev_optimistic:+,.2f}.\n"
        )

        if st.button("Generate Summary", key="bayes_summary_btn"):
            import requests as req

            if not OPENROUTER_API_KEY:
                st.error(
                    "OpenRouter API key not set. Add OPENROUTER_API_KEY to your environment, "
                    "or create a .env file with OPENROUTER_API_KEY=your_key. See .env.example."
                )
            else:
                with st.spinner("Generating summary..."):
                    try:
                        resp = req.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                "Content-Type": "application/json",
                            },
                        json={
                            "model": "openrouter/free",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a CRM experimentation analyst writing a Bayesian test summary "
                                        "that will be copy-pasted into a Slack message or email to CRM stakeholders.\n\n"
                                        "Format the output EXACTLY like this:\n\n"
                                        "**CRM Experiment Summary (Bayesian)**\n"
                                        "[One sentence describing the CRM experiment — use the experiment context provided]\n\n"
                                        "**Key Results**\n"
                                        "- Open Rate: [Control vs Variant, chance Variant is better, expected improvement]\n"
                                        "- Click Rate: [same]\n"
                                        "- Order Rate: [same]\n"
                                        "- Revenue Per User: [same — this is the bottom line metric]\n\n"
                                        "**Revenue Impact**\n"
                                        "[Worst case / most likely / best case revenue if rolled out to the full audience]\n\n"
                                        "**Verdict:** [Confident to roll out / Promising but needs more data / Too close to call / Stick with Control]\n\n"
                                        "Rules:\n"
                                        "- Frame everything in the context of CRM experimentation (email campaigns, lifecycle, engagement)\n"
                                        "- Use plain marketing language, no statistical jargon\n"
                                        "- Say 'X% chance Variant is better' not 'probability' or 'posterior'\n"
                                        "- Highlight the Revenue Per User metric — explain that it captures both conversion and order value\n"
                                        "- Include actual numbers and percentages\n"
                                        "- Keep it concise\n"
                                        "- Do NOT include a word count\n"
                                        "- Do NOT include a recommendation section beyond the verdict\n"
                                        "- Use GBP (£) for currency"
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"Experiment context: {experiment_description or 'Not provided'}\n\n"
                                        f"Write a Bayesian test summary for these results:\n\n{bayes_summary_context}"
                                    ),
                                },
                            ],
                        },
                        timeout=30,
                    )
                        resp.raise_for_status()
                        summary = resp.json()["choices"][0]["message"]["content"]
                        st.markdown("---")
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Could not generate summary: {e}")

    # -------------------------------------------------------------------
    # Step 6: Show workings for revenue scenarios
    # -------------------------------------------------------------------

    with st.expander("Step 6 — How We Calculated the Revenue Scenarios"):
        st.markdown(
            "Here's a full breakdown of how the **worst case**, **most likely**, and **best case** "
            "revenue figures are derived from the Bayesian simulation."
        )

        st.markdown("##### Input Data")
        st.markdown(
            f"""
| | Control | Variant |
|---|---|---|
| **Total Sends** | {sample_a:,} | {sample_b:,} |
| **Orders** | {orders_a:,} | {orders_b:,} |
| **Revenue** | £{revenue_a:,.2f} | £{revenue_b:,.2f} |
| **Revenue Per User (RPU)** | £{bayes_rpu['rpu_a']:.2f} | £{bayes_rpu['rpu_b']:.2f} |
"""
        )

        st.markdown("##### How the Simulation Works")
        st.markdown(
            f"""
1. We simulate **10,000 versions** of the test, each time drawing a plausible RPU for Control and Variant.
2. For each simulation we calculate **Variant RPU − Control RPU** (the "lift").
3. We then line up all 10,000 lifts from smallest to largest and read off three key numbers:
   - The **average lift** → the **most likely** scenario.
   - The **2.5th percentile** → the **worst case** (only 2.5% of simulations were worse).
   - The **97.5th percentile** → the **best case** (only 2.5% of simulations were better).
4. To turn each RPU lift into a revenue number, we multiply by **total sends** ({total_sends:,}).
"""
        )

        st.markdown("##### RPU Calculation")
        st.markdown(
            f"""
| | Revenue | ÷ Sends | = RPU |
|---|---|---|---|
| **Control** | £{revenue_a:,.2f} | ÷ {sample_a:,} | = **£{bayes_rpu['rpu_a']:.2f}** per user |
| **Variant** | £{revenue_b:,.2f} | ÷ {sample_b:,} | = **£{bayes_rpu['rpu_b']:.2f}** per user |

**Observed RPU difference** = £{bayes_rpu['rpu_b']:.2f} − £{bayes_rpu['rpu_a']:.2f} = **£{bayes_rpu['rpu_b'] - bayes_rpu['rpu_a']:+.2f}** per user
"""
        )

        st.markdown("##### Simulation Results → Revenue Scenarios")
        st.markdown(
            "After running 10,000 simulations to account for uncertainty, here are the three scenarios:"
        )
        st.markdown(
            f"""
| Scenario | How it's derived | RPU Lift per User | × Total Sends ({total_sends:,}) | = Revenue Impact |
|---|---|---|---|---|
| **Worst case** | 2.5th percentile of simulated lifts | £{bayes_rpu['ci_low']:+.2f} | × {total_sends:,} | **£{rpu_rev_pessimistic:+,.2f}** |
| **Most likely** | Average of simulated lifts | £{bayes_rpu['expected_lift']:+.2f} | × {total_sends:,} | **£{rpu_rev_expected:+,.2f}** |
| **Best case** | 97.5th percentile of simulated lifts | £{bayes_rpu['ci_high']:+.2f} | × {total_sends:,} | **£{rpu_rev_optimistic:+,.2f}** |
"""
        )

        st.markdown("##### Worked Example (Most Likely)")
        st.markdown(
            f"""
```
Most likely RPU lift   = £{bayes_rpu['expected_lift']:+.2f} per user
Total sends            = {total_sends:,}

Revenue impact         = £{bayes_rpu['expected_lift']:+.2f} × {total_sends:,}
                       = £{rpu_rev_expected:+,.2f}
```
"""
        )

        st.info(
            f"**Reading the numbers:** {fmt_prob(bayes_rpu['prob_b_better'])} of simulations "
            f"had Variant RPU higher than Control. The most likely revenue impact is "
            f"**£{rpu_rev_expected:+,.2f}**, but it could realistically range from "
            f"**£{rpu_rev_pessimistic:+,.2f}** (worst case) to **£{rpu_rev_optimistic:+,.2f}** (best case).",
            icon="💡",
        )
