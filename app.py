"""
Data Analysis Toolkit — SRH Tools & Methods of Data Analysis
Covers: Exploration · Descriptive Stats · Visualization ·
        Probability Distributions · Statistical Inference
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import (
        shapiro, kstest, anderson, chisquare,
        ttest_1samp, ttest_ind, ttest_rel,
        f_oneway, kruskal, levene,
        norm, binom, poisson, expon,
        uniform as sp_uniform, bernoulli,
        chi2, mannwhitneyu, wilcoxon, ranksums,
        lognorm, gamma, beta,
    )
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import io

# AI Assistant & system monitoring
import json
import subprocess
import platform
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analysis Toolkit",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 0.78rem; }
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
# -- Data generation for demo purposes --
def generate_sample_data():
    """Generates a synthetic retail dataset for demo purposes."""
    np.random.seed(42)
    n = 200
    
    # Generate synthetic data
    data = {
        "CustomerID": range(1, n + 1),
        "Age": np.random.randint(18, 70, size=n),
        "Annual_Income_k": np.random.lognormal(mean=3.8, sigma=0.3, size=n).round(1) * 10,        "Spending_Score": np.random.randint(1, 100, size=n),
        "Membership_Status": np.random.choice(["Basic", "Premium", "VIP"], size=n, p=[0.5, 0.3, 0.2]),
        "Region": np.random.choice(["North", "South", "East", "West"], size=n),
        "Purchase_Frequency": np.random.poisson(lam=10, size=n),
        "Satisfaction_Score": np.random.normal(loc=7, scale=1.5, size=n).clip(1, 10)
    }
    # Helper for log conversion
    def ln(x): return np.log(x)

    df = pd.DataFrame(data)
    
    # Create some correlations for the regression tool to find something interesting
    # Make Spending_Score correlate with Annual_Income_k and Membership_Status
    df['Spending_Score'] = (0.4 * df['Annual_Income_k'] + np.random.normal(0, 10, n)).clip(1, 100).astype(int)
        
    return df


def normality_table(series: pd.Series) -> pd.DataFrame:
    rows = []
    s = series.dropna()
    if len(s) >= 3:
        if len(s) <= 5000:
            stat, p = shapiro(s)
            rows.append({"Test": "Shapiro-Wilk", "Statistic": f"{stat:.4f}",
                         "p-value": f"{p:.4f}", "Normal (p>0.05)?": "Yes [OK]" if p > 0.05 else "No [!]"})
        s_std = (s - s.mean()) / s.std()
        stat, p = kstest(s_std, "norm")
        rows.append({"Test": "Kolmogorov-Smirnov", "Statistic": f"{stat:.4f}",
                     "p-value": f"{p:.4f}", "Normal (p>0.05)?": "Yes [OK]" if p > 0.05 else "No [!]"})
        ad = anderson(s, dist="norm")
        # Anderson-Darling critical value at 5% significance
        ad_cv = ad.critical_values[2]  # index 2 = 5%
        ad_normal = "Yes [OK]" if ad.statistic < ad_cv else "No [!]"
        rows.append({"Test": "Anderson-Darling", "Statistic": f"{ad.statistic:.4f}",
                     "p-value": f"crit@5%={ad_cv:.4f}", "Normal (p>0.05)?": ad_normal})
    return pd.DataFrame(rows)


def p_stars(p_value, num=10):
    """Convert p-value to star representation."""
    if p_value <= 0:
        stars = "*******"
    else:
        stars = "*" * int(-np.log10(p_value)) if p_value > 0 else "*******"
        stars = stars[:num].ljust(num, ".")
    return stars

def nice_fig(figsize=(8, 6)):
    """Create a properly sized and styled matplotlib figure."""
    # Handle both tuple and separate args
    if isinstance(figsize, tuple):
        w, h = figsize
    else:
        w, h = figsize, 6
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.tight_layout()
    return fig, ax


# ── AI Assistant helpers ──────────────────────────────────────────────────────
def get_ollama_models():
    """Return a list of locally available Ollama model names."""
    if not _HAS_OLLAMA:
        return []
    try:
        resp = ollama.list()
        models = []
        for m in resp.get("models", []):
            name = m.get("model") or m.get("name")
            if name:
                models.append(name)
        return models
    except Exception:
        return []


def get_system_stats():
    """Return CPU, RAM and GPU usage stats for the local machine."""
    stats_out = {
        "cpu_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_percent": None,
        "gpu_info": "N/A",
    }
    if _HAS_PSUTIL:
        try:
            stats_out["cpu_percent"] = psutil.cpu_percent(interval=0.3)
            vm = psutil.virtual_memory()
            stats_out["ram_used_gb"] = vm.used / (1024 ** 3)
            stats_out["ram_total_gb"] = vm.total / (1024 ** 3)
            stats_out["ram_percent"] = vm.percent
        except Exception:
            pass

    # GPU info - platform specific
    try:
        if platform.system() == "Darwin":
            # Apple Silicon: unified memory; report chip name
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2
            )
            chip = result.stdout.strip()
            if not chip:
                # Try hardware model
                result = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True, text=True, timeout=2
                )
                chip = result.stdout.strip()
            stats_out["gpu_info"] = f"{chip} (unified memory GPU)" if chip else "Apple GPU (unified memory)"
        else:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                util = parts[0].strip()
                mem_used = parts[1].strip()
                mem_total = parts[2].strip()
                stats_out["gpu_info"] = f"{util}% util · {mem_used}/{mem_total} MB"
    except Exception:
        pass

    return stats_out


def get_dataframe_context(df):
    """Build a concise text summary of the loaded dataframe for the AI."""
    if df is None:
        return "No dataset is currently loaded."
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    lines = [
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns.",
        f"Numeric columns ({len(num_cols)}): {', '.join(num_cols) if num_cols else 'none'}",
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'none'}",
        f"Missing values total: {int(df.isnull().sum().sum())}",
    ]
    if num_cols:
        desc = df[num_cols].describe().round(3)
        lines.append("Summary statistics (numeric columns):")
        lines.append(desc.to_string())
    return "\n".join(lines)


# ── AI memory persistence ─────────────────────────────────────────────────────
import os as _os

AI_MEMORY_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ai_memory.json")


def load_ai_memory():
    """Load persistent AI memory notes from disk."""
    try:
        if _os.path.exists(AI_MEMORY_FILE):
            with open(AI_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_ai_memory(memory_list):
    """Persist AI memory notes to disk."""
    try:
        with open(AI_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_list, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# ── Token estimation & context helpers ────────────────────────────────────────
def estimate_tokens(text):
    """Rough token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, int(len(str(text)) / 4))


def estimate_messages_tokens(messages):
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", ""))
        if m.get("thinking"):
            total += estimate_tokens(m.get("thinking", ""))
    return total


def summarize_conversation(model, messages, options):
    """Use the AI to summarize older messages into a compact note."""
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m.get('content', '')}"
        for m in messages if m["role"] in ("user", "assistant")
    )
    prompt = [
        {"role": "system", "content": "You summarize conversations concisely, preserving "
                                       "key facts, decisions, and findings. Output a short bullet summary."},
        {"role": "user", "content": "Summarize this conversation so it can replace the full "
                                     "history while keeping important context:\n\n" + convo_text},
    ]
    resp = ollama.chat(model=model, messages=prompt, options=options)
    return resp["message"]["content"]


# ── AI agent tools ────────────────────────────────────────────────────────────
def ai_tool_describe(df, column=None):
    """Compute descriptive statistics for a column or whole dataset."""
    if df is None:
        return "No dataset loaded."
    if column and column in df.columns:
        s = df[column].dropna()
        if pd.api.types.is_numeric_dtype(s):
            return s.describe().round(4).to_string()
        return s.value_counts().head(20).to_string()
    return df.describe(include="all").round(4).to_string()


def ai_tool_correlation(df, col1=None, col2=None):
    """Compute correlation between two columns or full correlation matrix."""
    if df is None:
        return "No dataset loaded."
    num = df.select_dtypes(include=np.number)
    if col1 and col2 and col1 in num.columns and col2 in num.columns:
        r = num[col1].corr(num[col2])
        return f"Pearson correlation between {col1} and {col2}: {r:.4f}"
    return num.corr().round(3).to_string()


def ai_tool_ttest(df, column, group_column):
    """Run a two-sample t-test on `column` grouped by `group_column`."""
    if df is None:
        return "No dataset loaded."
    if column not in df.columns or group_column not in df.columns:
        return f"Columns not found. Available: {', '.join(df.columns)}"
    groups = df[group_column].dropna().unique()
    if len(groups) < 2:
        return "Grouping column needs at least 2 groups."
    g1 = df[df[group_column] == groups[0]][column].dropna()
    g2 = df[df[group_column] == groups[1]][column].dropna()
    t, p = ttest_ind(g1, g2, equal_var=False)
    return (f"Welch t-test of {column} between {groups[0]} and {groups[1]}: "
            f"t={t:.4f}, p={p:.4f}. "
            f"{'Significant difference' if p < 0.05 else 'No significant difference'} (alpha=0.05).")


def ai_tool_value_counts(df, column):
    """Return value counts for a column."""
    if df is None:
        return "No dataset loaded."
    if column not in df.columns:
        return f"Column not found. Available: {', '.join(df.columns)}"
    return df[column].value_counts().head(20).to_string()


def ai_tool_missing(df):
    """Report missing values per column."""
    if df is None:
        return "No dataset loaded."
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        return "No missing values in the dataset."
    return miss.to_string()


def ai_tool_remember(content, category="note"):
    """Store a note in persistent AI memory."""
    if not content or not str(content).strip():
        return "Cannot store empty memory."
    memory = st.session_state.get("ai_memory", [])
    entry = {
        "content": str(content).strip(),
        "category": str(category or "note"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    memory.append(entry)
    st.session_state.ai_memory = memory
    save_ai_memory(memory)
    return f"Stored memory ({category}): {content}"


def ai_tool_recall(query=None):
    """Retrieve notes from persistent AI memory, optionally filtered by query."""
    memory = st.session_state.get("ai_memory", [])
    if not memory:
        return "Memory is empty."
    if query:
        q = str(query).lower()
        hits = [m for m in memory if q in m["content"].lower() or q in m["category"].lower()]
        if not hits:
            return f"No memories matching '{query}'."
        memory = hits
    return "\n".join(f"[{m['category']}] {m['content']} (saved {m['timestamp']})" for m in memory)


def ai_tool_add_to_report(df, title, content):
    """Add a text/result item to the PDF report builder."""
    items = st.session_state.get("report_items", [])
    items.append({
        "type": "custom_note",
        "title": str(title or "AI Analysis"),
        "content": str(content or ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    st.session_state.report_items = items
    return f"Added '{title}' to the report. Report now has {len(items)} item(s)."


def _fig_to_png_bytes(fig):
    """Render a matplotlib figure to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def ai_tool_plot(df, kind, column=None, col2=None, group_column=None, bins=30):
    """Generate a plot from the dataset and queue its image for the chat.

    kind: histogram | box | scatter | bar | correlation | line
    Returns a short text confirmation; the image is stored in session state
    and rendered in the chat by the AI Assistant view.
    """
    if df is None:
        return "No dataset loaded, cannot plot."
    kind = (kind or "").strip().lower()
    try:
        if kind == "histogram":
            if not column or column not in df.columns:
                return f"Column '{column}' not found for histogram."
            data = pd.to_numeric(df[column], errors="coerce").dropna()
            fig, ax = nice_fig((7, 4))
            ax.hist(data, bins=int(bins or 30), color="#4c9be8", edgecolor="white", alpha=0.85)
            ax.set_xlabel(column); ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram - {column}")
            caption = f"Histogram of {column}"

        elif kind == "box":
            cols = [c for c in [column, col2] if c and c in df.columns]
            if not cols:
                return "No valid numeric columns for box plot."
            fig, ax = nice_fig((7, 4))
            ax.boxplot([pd.to_numeric(df[c], errors="coerce").dropna() for c in cols],
                       tick_labels=cols)
            ax.set_ylabel("Value"); ax.set_title("Box Plot")
            caption = f"Box plot of {', '.join(cols)}"

        elif kind == "scatter":
            if not column or not col2 or column not in df.columns or col2 not in df.columns:
                return "Scatter plot needs two valid numeric columns (column, col2)."
            x = pd.to_numeric(df[column], errors="coerce")
            y = pd.to_numeric(df[col2], errors="coerce")
            mask = x.notna() & y.notna()
            fig, ax = nice_fig((7, 5))
            ax.scatter(x[mask], y[mask], color="#4c9be8", s=18, alpha=0.6)
            ax.set_xlabel(column); ax.set_ylabel(col2)
            ax.set_title(f"Scatter - {column} vs {col2}")
            caption = f"Scatter plot of {column} vs {col2}"

        elif kind == "bar":
            if not column or column not in df.columns:
                return f"Column '{column}' not found for bar chart."
            vc = df[column].value_counts().head(20)
            fig, ax = nice_fig((7, 4))
            ax.bar(vc.index.astype(str), vc.values, color="#4c9be8", edgecolor="white")
            ax.set_xlabel(column); ax.set_ylabel("Count")
            ax.set_title(f"Value counts - {column}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            caption = f"Bar chart of {column}"

        elif kind == "line":
            if not column or column not in df.columns:
                return f"Column '{column}' not found for line chart."
            data = pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)
            fig, ax = nice_fig((7, 4))
            ax.plot(data.index, data.values, color="#4c9be8")
            ax.set_xlabel("Index"); ax.set_ylabel(column)
            ax.set_title(f"Line chart - {column}")
            caption = f"Line chart of {column}"

        elif kind == "correlation":
            num = df.select_dtypes(include=np.number)
            if num.shape[1] < 2:
                return "Need at least 2 numeric columns for a correlation heatmap."
            corr = num.corr()
            fig, ax = nice_fig((6, 5))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(corr.index)))
            ax.set_yticklabels(corr.index)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Correlation heatmap")
            caption = "Correlation heatmap"
        else:
            return (f"Unknown plot kind '{kind}'. Use one of: "
                    "histogram, box, scatter, bar, line, correlation.")

        png = _fig_to_png_bytes(fig)
        st.session_state.setdefault("_ai_plot_images", []).append(
            {"caption": caption, "data": png}
        )
        return f"Created a {kind} plot ({caption}). It is shown in the chat."
    except Exception as e:
        return f"Failed to create {kind} plot: {e}"


# Map tool names to functions for the agent
AI_TOOLS = {
    "describe": ai_tool_describe,
    "correlation": ai_tool_correlation,
    "ttest": ai_tool_ttest,
    "value_counts": ai_tool_value_counts,
    "missing_values": ai_tool_missing,
    "remember": ai_tool_remember,
    "recall": ai_tool_recall,
    "add_to_report": ai_tool_add_to_report,
    "plot": ai_tool_plot,
}

AI_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "describe",
            "description": "Get descriptive statistics (mean, std, quartiles, etc.) for a column or the whole dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name. Omit for whole-dataset summary."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlation",
            "description": "Compute Pearson correlation between two numeric columns, or the full correlation matrix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First numeric column."},
                    "col2": {"type": "string", "description": "Second numeric column."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ttest",
            "description": "Run a two-sample t-test of a numeric column grouped by a categorical column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Numeric column to test."},
                    "group_column": {"type": "string", "description": "Categorical column defining 2 groups."},
                },
                "required": ["column", "group_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "value_counts",
            "description": "Get the frequency counts of unique values in a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name."},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "missing_values",
            "description": "Report the number of missing values per column.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save an important fact, user preference, or finding to persistent memory for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to remember."},
                    "category": {"type": "string", "description": "Optional category, e.g. 'preference', 'finding', 'note'."},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Retrieve previously stored memory notes, optionally filtered by a search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional search term. Omit to list all memories."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_report",
            "description": "Add an analysis result, summary, or note to the PDF report builder for later export.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Section title for the report item."},
                    "content": {"type": "string", "description": "The text/result content to include."},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot",
            "description": "Create a chart from the dataset and display it as an image in the chat. "
                           "Use this whenever the user asks to see, show, visualize, or plot data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string",
                              "description": "Chart type: histogram, box, scatter, bar, line, or correlation."},
                    "column": {"type": "string", "description": "Primary column (x / value)."},
                    "col2": {"type": "string", "description": "Second column (y for scatter, optional for box)."},
                    "group_column": {"type": "string", "description": "Optional grouping column."},
                    "bins": {"type": "integer", "description": "Number of bins for a histogram."},
                },
                "required": ["kind"],
            },
        },
    },
]


def run_ai_tool(name, args, df):
    """Dispatch an AI tool call to the corresponding function."""
    func = AI_TOOLS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        if name == "describe":
            return func(df, args.get("column"))
        if name == "correlation":
            return func(df, args.get("col1"), args.get("col2"))
        if name == "ttest":
            return func(df, args.get("column"), args.get("group_column"))
        if name == "value_counts":
            return func(df, args.get("column"))
        if name == "missing_values":
            return func(df)
        if name == "remember":
            return func(args.get("content"), args.get("category", "note"))
        if name == "recall":
            return func(args.get("query"))
        if name == "add_to_report":
            return func(df, args.get("title"), args.get("content"))
        if name == "plot":
            return func(df, args.get("kind"), args.get("column"), args.get("col2"),
                        args.get("group_column"), args.get("bins", 30))
    except Exception as e:
        return f"Error running tool {name}: {e}"
    return "No result."


# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
if "report_items" not in st.session_state:
    st.session_state.report_items = []
if "report_title" not in st.session_state:
    st.session_state.report_title = "Data Analysis Report"
if "report_author" not in st.session_state:
    st.session_state.report_author = "Data Analyst"
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "ai_model" not in st.session_state:
    st.session_state.ai_model = None
if "ai_memory" not in st.session_state:
    st.session_state.ai_memory = load_ai_memory()
if "ai_context_limit" not in st.session_state:
    st.session_state.ai_context_limit = 4096

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Data Analysis Toolkit")
    
    with st.expander("🗺️ Sidebar Map", expanded=False):
        st.markdown("""
        **Quick Guide to this Sidebar:**
        - 🧭 **Navigate**: Switch between analysis tools and modules.
        - 💾 **Data Loading**: Upload your CSV or load sample data to start.
        - 📋 **Report Builder**: Track and manage items added to your final PDF.
        """)
    
    st.markdown("---")
    TOOLS = [
        "Home",
        "Data Explorer",
        "Descriptive Statistics",
        "Data Visualization",
        "Probability Distributions",
        "CLT Demonstration",
        "Statistical Inference",
        "Regression Analysis",
        "Confidence Intervals",
        "Non-Parametric Tests",
        "Data Transformations",
        "Sampling Methods",
        "Generate PDF Report",
        "AI Assistant",
    ]
    tool = st.radio("Navigate",TOOLS, label_visibility="collapsed")
    st.markdown("---")

    st.markdown("### :material/database: Data Loading")
    
    # Sidebar Dataset Upload
    uploaded_sidebar = st.file_uploader("Upload a CSV file", type=["csv"], key="sidebar_uploader")
    if uploaded_sidebar is not None:
        try:
            st.session_state["df"] = pd.read_csv(uploaded_sidebar)
            st.success("Dataset loaded successfully!")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Load Sample Data", use_container_width=True):
            st.session_state["df"] = generate_sample_data()
            st.success("Sample loaded!")
            st.rerun()
    
    # Show current dataset status in sidebar
    df_current = st.session_state.get("df")
    if df_current is not None:
        st.markdown(f"**Active Dataset:** {df_current.shape[0]} rows, {df_current.shape[1]} cols")
        if st.button(":material/delete: Clear Dataset", use_container_width=True):
            st.session_state["df"] = None
            st.rerun()
    else:
        st.info("No dataset loaded")

    st.markdown("---")
    # Report Builder Status
    st.markdown("### :material/content_paste: Report Builder")
    n_items = len(st.session_state.report_items)
    if n_items > 0:
        st.success(f":material/check_circle: {n_items} item(s) in report")
        if st.button(":material/delete: Clear All Items"):
            st.session_state.report_items = []
            st.rerun()
    else:
        st.info("No items added yet")
    st.caption("Use 'Add to Report' buttons in each tool to build your report.")

    st.markdown("---")
    st.caption("SRH · Tools & Methods of Data Analysis")
tool_key = tool

# ════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════
# --- HOME PAGE ---
page = tool if 'choice' in locals() else (menu if 'menu' in locals() else "Home")
if tool == "Home":
    st.title(":material/analytics: Data Analysis Toolkit")
    st.subheader("Your All-in-One Workspace for Statistical Analysis & Data Science")

    with st.container(border=True):
        st.markdown("### :material/map: Quick Workspace Map")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown("**1. Setup**\n- Upload CSV\n- Load Samples\n- Clean Data")
        with m_col2:
            st.markdown("**2. Analyze**\n- Stats & Visuals\n- Probability\n- Inference")
        with m_col3:
            st.markdown("**3. Export**\n- AI Synthesis\n- PDF Report\n- CSV Export")

    st.markdown("""
    Welcome to the Data Analysis Toolkit. This application is designed to streamline your entire data workflow from initial raw file exploration to advanced statistical hypothesis testing and automated report generation.
    """)

    st.markdown("### :material/menu_book: Documentation & User Guide")
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["🚀 Quick Start", "📚 Methodology", "🛠️ Troubleshooting"])
    
    with doc_tab1:
        st.markdown("""
        **How to get started in 3 steps:**
        1. **Load Data**: Use the sidebar uploader to import your `.csv` file or click 'Load Sample Data'.
        2. **Explore & Clean**: Visit the **Data Explorer** to check for missing values and clean your dataset.
        3. **Analyze**: Select any tool from the 'Navigate' menu (e.g., *Descriptive Statistics* or *Regression Analysis*) to generate insights.
        """)
        
    with doc_tab2:
        st.markdown("""
        **Statistical Rigor:**
        - **Parametric vs Non-Parametric**: We provide automated checks. If the *Shapiro-Wilk* test fails (p < 0.05), we recommend using the Non-Parametric alternatives (like *Mann-Whitney U*).
        - **ClT**: The Central Limit Theorem tool demonstrates how sampling distributions converge to normality as sample size increases.
        - **Regression**: Our models include $R^2$ and MSE metrics to validate the goodness-of-fit.
        """)
        
    with doc_tab3:
        st.markdown("""
        **Common Issues:**
        - **File Error**: Ensure your CSV is properly encoded (UTF-8) and doesn't have merged cells.
        - **Empty Analysis**: Most tools require an active dataset. If you see "No dataset loaded", check the sidebar.
        - **Report Empty**: Remember to click the **'Add to Report'** button inside each tool before going to the *Generate PDF Report* page.
        """)

    st.write("---")

    # --- SECTION 1: KEY FEATURES GRID ---
    st.markdown("## :material/star: Key Features & Core Modules")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("### :material/search: 1. Data Exploration & Prep")
            st.write("""
            * **Smart Imputation:** Handle missing data using Mean, Median, Mode, or dropping rows.
            * **Data Profiling:** Inspect data types, shapes, and detect anomalies at a glance.
            * **Transforms:** Easily apply Log, Square Root, or Min-Max scaling to features.
            """)
            
        with st.container(border=True):
            st.markdown("### :material/bar_chart: 2. Descriptive Stats & Visuals")
            st.write("""
            * **Moment Metrics:** Compute Mean, Median, Variance, Skewness, and Kurtosis.
            * **Interactive Plots:** Generate dynamic Box plots, Histograms, and Scatter matrices.
            * **Correlation Engine:** Map feature relationships instantly via Correlation Heatmaps.
            """)

        with st.container(border=True):
            st.markdown("### :material/functions: 3. Probability Distributions")
            st.write("""
            * **Discrete & Continuous:** Model data against Normal, Binomial, Poisson, and Exponential models.
            * **Fit Testing:** Check matching data criteria using PDF, CDF, and PPF calculations.
            """)

    with col2:
        with st.container(border=True):
            st.markdown("### :material/science: 4. Statistical Inference Engine")
            st.write("""
            * **Parametric Testing:** Run 1-Sample/Independent/Paired t-tests and One-Way ANOVA.
            * **Non-Parametric Fallbacks:** Swap safely to Mann-Whitney, Wilcoxon, or Kruskal-Wallis.
            * **Assumption Checks:** Automated Shapiro-Wilk and Levene variance validations.
            """)
            
        with st.container(border=True):
            st.markdown("### :material/trending_up: 5. Predictive Modeling & Regression")
            st.write("""
            * **Linear Fits:** Compute Simple and Multiple Linear Regressions.
            * **Performance Diagnostics:** Review R-squared, Mean Squared Error (MSE), and Residual breakdowns.
            """)

        with st.container(border=True):
            st.markdown("### :material/smart_toy: 6. AI Agent & PDF Reports")
            st.write("""
            * **Ollama AI Assistant:** Chat naturally with a local LLM to query insights directly from your data.
            * **Executive PDF Generator:** Compile your active session's tables and metrics into a publication-ready PDF report.
            """)

    st.write("---")

    # --- SECTION 2: NAVIGATION MAP & PAGE DIRECTORY ---
    st.markdown("## :material/explore: Workspace Navigation Directory")
    st.markdown("Use the main sidebar menu selector to navigate across these specialized tools:")

    # Create 3 mini columns to list all available tools from your app configuration
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("**Core Operations**")
        st.write("""
        * `:material/home: Home` — Overview dashboard & quickstart guidelines.
        * `:material/database: Data Explorer` — File parsing, filtering, and row adjustments.
        * `:material/calculate: Descriptive Statistics` — Instant central tendency & shape metrics.
        * `:material/insert_chart: Data Visualization` — Interactive distribution & scatter graphs.
        """)
        
    with nav_col2:
        st.markdown("**Advanced Calculations**")
        st.write("""
        * `:material/timeline: Transformations` — Mathematical scaling (Log, Square root, Z-score).
        * `:material/show_chart: Probability Distributions` — Model distributions & compute probabilities.
        * `:material/experiment: Hypothesis Testing` — Mean differences & distribution checks.
        * `:material/monitoring: Regression Analysis` — Predictive line fitting and validation.
        """)
        
    with nav_col3:
        st.markdown("**Utilities & Outputs**")
        st.write("""
        * `:material/layers: Sampling & Simulation` — Generate subsets or random data arrays.
        * `:material/chat: AI Assistant` — Native Ollama LLM execution on local schemas.
        * `:material/picture_as_pdf: Generate PDF Report` — Document creation with active artifacts.
        """)

    st.write("---")

    # --- SECTION 3: REAL-WORLD USE CASES ---
    st.markdown("## :material/work: Practical Use Cases")
    
    tab1, tab2, tab3 = st.tabs([
        "🔬 Academic Research & Lab Work", 
        "📈 Business Intelligence & Marketing", 
        "🏥 Health & Clinical Metrics"
    ])
    
    with tab1:
        st.write("""
        * **Check Assumptions First:** Load sample experiments into the **Data Explorer**, navigate to **Hypothesis Testing**, and run *Shapiro-Wilk* tests instantly to choose between a Parametric *t-test* or Non-Parametric *Mann-Whitney U* test.
        * **Formula Testing:** Use **Probability Distributions** to compare mathematical curve expectations directly with captured field variables.
        """)
        
    with tab2:
        st.write("""
        * **Trend Mapping:** Run **Regression Analysis** on ad campaign metrics to compute your $R^2$ performance indicator and evaluate coefficients for predictable scaling models.
        * **Fast Executive Reporting:** Once correlations are identified, drop down to **Generate PDF Report** to produce an institutional record ready for executive distribution.
        """)
        
    with tab3:
        st.write("""
        * **Before/After Assessments:** Evaluate patient indicators across treatment checkpoints via **Paired t-tests** or **Wilcoxon Signed-Rank** methods.
        * **Data Cleaning & Rescaling:** Handle incomplete diagnostic panels seamlessly via the **Data Explorer's** automated mean/median missing data imputation tools.
        """)

    st.write("---")
    
    st.info(":material/lightbulb: **Getting Started:** Drop your `.csv` or `.xlsx` file into the sidebar uploader to unlock all analysis modules.")
# ══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Data Explorer":
    st.header("Data Explorer",
              help="Upload a CSV, preview rows, inspect data types and missing values, "
                   "detect outliers (IQR), clean/impute data, and export the result.")

    df = st.session_state.get("df")
    if df is None:
        st.info("No dataset loaded. Please upload a CSV or load sample data from the sidebar to begin.")
        st.stop()

    # Overview
    st.subheader("Dataset Overview")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rows", df.shape[0])
    r2.metric("Columns", df.shape[1])
    r3.metric("Numeric cols", len(df.select_dtypes(include=np.number).columns))
    r4.metric("Categorical cols", len(df.select_dtypes(exclude=np.number).columns))

    with st.expander("Preview Data (first 50 rows)", expanded=True):
        st.dataframe(df.head(50), width="stretch")

    # Data types & missing values
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Column Data Types"):
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Dtype":  df.dtypes.astype(str).values,
                "Non-Null": df.count().values,
                "Unique": [df[c].nunique() for c in df.columns],
            })
            st.dataframe(dtype_df, width="stretch", hide_index=True)

    with col_b:
        with st.expander("Missing Values"):
            miss = df.isnull().sum()
            miss_pct = (miss / len(df) * 100).round(2)
            miss_df = pd.DataFrame({
                "Column": miss.index,
                "Missing": miss.values,
                "Missing %": miss_pct.values,
            })
            miss_df = miss_df[miss_df["Missing"] > 0]
            if miss_df.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(miss_df, width="stretch", hide_index=True)

    # Outlier detection
    with st.expander("Outlier Detection (IQR method)"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("No numeric columns.")
        else:
            sel_out = st.selectbox("Select column", num_cols, key="out_col")
            col_data = df[sel_out].dropna()
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower) | (col_data > upper)]
            st.write(f"IQR bounds: **[{lower:.2f}, {upper:.2f}]** — "
                     f"**{len(outliers)} outlier(s)** ({len(outliers)/len(col_data)*100:.1f}%)")
            fig, ax = nice_fig((7, 3))
            ax.boxplot(col_data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor="#cce5ff"),
                       medianprops=dict(color="navy", lw=2),
                       flierprops=dict(marker="o", color="red", alpha=0.5))
            ax.set_xlabel(sel_out)
            ax.set_title(f"Box Plot — {sel_out}")
            st.pyplot(fig, width="content")
            plt.close(fig)

    # Data cleaning - ENHANCED with Mode Imputation
    with st.expander("Data Cleaning — Handle Missing Values"):
        num_miss_cols = [c for c in df.columns if df[c].isnull().any() and pd.api.types.is_numeric_dtype(df[c])]
        cat_miss_info = []
        cat_clean_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].isnull().any()]

        for cat_col in cat_clean_cols:
            mode_val = df[cat_col].mode()
            if len(mode_val) > 0:
                cat_miss_info.append((cat_col, mode_val.iloc[0]))

        all_clean_candidates = num_miss_cols + [c[0] for c in cat_miss_info]

        if not all_clean_candidates:
            st.success("No missing values found — nothing to clean.")
        else:
            clean_col_label = st.selectbox("Select Column with Missing Values", all_clean_candidates, key="clean_col")

            # Determine if numeric or categorical for method display
            is_numeric = pd.api.types.is_numeric_dtype(df[clean_col_label])

            options_if_numeric = ["Drop rows at this column", "Fill with Mean", "Fill with Median", "Fill with 0"]
            
            # Fix the string representation of mode for the dropdown
            mode_val_for_label = "Unknown"
            for col, val in cat_miss_info:
                if col == clean_col_label:
                    mode_val_for_label = val
                    break
            options_if_categorical = ["Drop rows at this column", f"Fill with Mode ({mode_val_for_label})"]

            # Determine available methods
            current_options = options_if_categorical if not is_numeric and clean_col_label in cat_clean_cols else options_if_numeric
            
            clean_method = st.selectbox(
                "Imputation Strategy",
                current_options,
                key="clean_method"
            )

            if st.button("Apply Cleaning"):
                df2 = df.copy()
                applied_desc = ""

                if clean_method == "Drop rows at this column":
                    dropped_rows = len(df2) - df2.dropna(subset=[clean_col_label]).shape[0]
                    df2 = df2.dropna(subset=[clean_col_label])
                    applied_desc = f"Removed {dropped_rows} row(s)"

                elif clean_method == "Fill with Mean":
                    if is_numeric:
                        fill_val = df2[clean_col_label].mean()
                        prev_count = df2[clean_col_label].isna().sum()
                        df2[clean_col_label] = df2[clean_col_label].fillna(fill_val)
                        applied_desc = f"Filled {prev_count} missing value(s) with mean ({fill_val:.4f})"

                elif clean_method == "Fill with Median":
                    if is_numeric:
                        fill_val = df2[clean_col_label].median()
                        prev_count = df2[clean_col_label].isna().sum()
                        df2[clean_col_label] = df2[clean_col_label].fillna(fill_val)
                        applied_desc = f"Filled {prev_count} missing value(s) with median ({fill_val:.4f})"

                elif clean_method == "Fill with 0":
                    prev_count = df2[clean_col_label].isna().sum()
                    df2[clean_col_label] = df2[clean_col_label].fillna(0)
                    applied_desc = f"Filled {prev_count} missing value(s) with 0"

                elif "Fill with Mode" in clean_method:
                    if not is_numeric and clean_col_label in cat_clean_cols:
                        # Find the mode value from our pre-calculated list
                        mode_val = next((v for c, v in cat_miss_info if c == clean_col_label), None)
                        prev_count = df2[clean_col_label].isna().sum()
                        df2[clean_col_label] = df2[clean_col_label].fillna(mode_val)
                        applied_desc = f"Filled {prev_count} missing value(s) with mode ({mode_val})"

                st.session_state["df"] = df2
                st.success(f"✅ Applied: '{applied_desc}' to **{clean_col_label}**")
                st.rerun()

    # Export Dataset
    with st.expander("Export Dataset"):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        
        st.markdown("**Original Data:** (clicking this downloads the original unchanged CSV)")
        st.download_button(
            ":material/download: Download Original CSV",
            data=buf.getvalue(),
            file_name="dataset_original.csv",
            mime="text/csv"
        )

        st.markdown("**Note:** Apply cleaning operations in the 'Data Cleaning' section above first, then refresh this page to export your cleaned dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Descriptive Statistics":
    st.header("Descriptive Statistics",
              help="Summary table (mean, median, std, IQR, skew, kurtosis), per-column "
                   "deep-dive, and correlation/covariance matrices. Add results to your report.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found.")
        st.stop()

    sel_cols = st.multiselect(
        "Select columns", num_cols, default=num_cols[:min(5, len(num_cols))]
    )
    if not sel_cols:
        st.info("Select at least one column.")
        st.stop()

    sub = df[sel_cols]

    # Summary table
    st.subheader("Summary Table")
    rows = []
    for c in sel_cols:
        s = sub[c].dropna()
        rows.append({
            "Column":   c,
            "n":        len(s),
            "Mean":     round(s.mean(), 4),
            "Median":   round(s.median(), 4),
            "Mode":     round(s.mode().iloc[0], 4) if not s.mode().empty else "—",
            "Std Dev":  round(s.std(), 4),
            "Variance": round(s.var(), 4),
            "IQR":      round(s.quantile(0.75) - s.quantile(0.25), 4),
            "Min":      round(s.min(), 4),
            "Max":      round(s.max(), 4),
            "Skewness": round(s.skew(), 4),
            "Kurtosis": round(s.kurtosis(), 4),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Add to Report button
    if st.button(":material/content_paste: Add Summary Statistics to Report", key="add_desc_stats"):
        st.session_state.report_items.append({
            "type": "descriptive_stats",
            "title": "Descriptive Statistics Summary",
            "data": pd.DataFrame(rows).to_dict(),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        st.success(":material/check_circle: Added to report!")

    # Deep dive
    st.subheader("Column Deep-Dive")
    dc = st.selectbox("Column", sel_cols, key="dc_col")
    s = sub[dc].dropna()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Mean",     f"{s.mean():.4f}")
    m2.metric("Median",   f"{s.median():.4f}")
    m3.metric("Std Dev",  f"{s.std():.4f}")
    m4.metric("Variance", f"{s.var():.4f}")
    m5.metric("Skewness", f"{s.skew():.4f}")
    m6.metric("Kurtosis", f"{s.kurtosis():.4f}")

    skew_v = s.skew()
    st.caption("**Skewness:** " + (
        "Approximately symmetric" if abs(skew_v) < 0.5
        else ("Right-skewed (positive)" if skew_v > 0 else "Left-skewed (negative)")
    ))
    kurt_v = s.kurtosis()
    st.caption("**Kurtosis:** " + (
        "Mesokurtic (normal-like)" if abs(kurt_v) < 0.5
        else ("Leptokurtic (heavy tails)" if kurt_v > 0 else "Platykurtic (light tails)")
    ))

    # Correlation
    if len(sel_cols) > 1:
        st.subheader("Correlation & Covariance")
        t1, t2 = st.tabs(["Pearson Correlation Matrix", "Covariance Matrix"])
        with t1:
            corr = sub.corr(method="pearson")
            n = len(sel_cols)
            fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(n)); ax.set_xticklabels(sel_cols, rotation=45, ha="right")
            ax.set_yticks(range(n)); ax.set_yticklabels(sel_cols)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if abs(corr.values[i,j]) > 0.6 else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Pearson Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig, width="content")
            plt.close(fig)
            
            # Add to Report button
            if st.button(":material/content_paste: Add Correlation Matrix to Report", key="add_corr_matrix"):
                # Recreate figure for report
                fig_report, ax_report = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
                im_report = ax_report.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
                ax_report.set_xticks(range(n))
                ax_report.set_xticklabels(sel_cols, rotation=45, ha="right")
                ax_report.set_yticks(range(n))
                ax_report.set_yticklabels(sel_cols)
                for i in range(n):
                    for j in range(n):
                        ax_report.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                                fontsize=9, color="white" if abs(corr.values[i,j]) > 0.6 else "black")
                plt.colorbar(im_report, ax=ax_report, shrink=0.8)
                ax_report.set_title("Pearson Correlation Matrix")
                plt.tight_layout()
                
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig_report)
                img_buffer.seek(0)
                
                st.session_state.report_items.append({
                    "type": "correlation_heatmap",
                    "title": "Correlation Matrix Heatmap",
                    "image": img_buffer.getvalue(),
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                st.success(":material/check_circle: Added correlation heatmap to report!")
        
        with t2:
            st.dataframe(sub.cov().round(4), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# DATA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Data Visualization":
    st.header("Data Visualization",
              help="Build histograms, box plots, scatter plots, bar/line charts, Q-Q plots "
                   "and correlation heatmaps from your dataset's columns.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    chart = st.selectbox("Chart type", [
        "Histogram", "Box Plot", "Scatter Plot",
        "Bar Chart", "Line Chart", "Q-Q Plot", "Correlation Heatmap",
    ])

    if chart == "Histogram":
        col = st.selectbox("Column", num_cols)
        bins = st.slider("Bins", 5, 100, 30)
        kde = st.checkbox("KDE smoothing curve", value=True)
        data = df[col].dropna()
        fig, ax = nice_fig((8, 4))
        ax.hist(data, bins=bins, color="#4c9be8", edgecolor="white", alpha=0.85, density=kde)
        if kde:
            xs = np.linspace(data.min(), data.max(), 300)
            kernel = stats.gaussian_kde(data)
            ax2 = ax.twinx()
            ax2.plot(xs, kernel(xs), "crimson", lw=2, label="KDE")
            ax2.set_ylabel("Density")
            ax2.spines[["top"]].set_visible(False)
        ax.set_xlabel(col); ax.set_title(f"Histogram — {col}")
        st.pyplot(fig, width="content"); plt.close(fig)

    elif chart == "Box Plot":
        cols = st.multiselect("Columns", num_cols, default=num_cols[:min(4, len(num_cols))])
        if cols:
            fig, ax = nice_fig((max(6, len(cols) * 1.5), 5))
            data_list = [df[c].dropna().values for c in cols]
            bp = ax.boxplot(data_list, patch_artist=True,
                            medianprops=dict(color="navy", lw=2))
            cmap = plt.cm.Set2(np.linspace(0, 1, len(cols)))
            for patch, color in zip(bp["boxes"], cmap):
                patch.set_facecolor(color)
            ax.set_xticklabels(cols, rotation=30, ha="right")
            ax.set_title("Box Plot"); ax.set_ylabel("Value")
            st.pyplot(fig, width="content"); plt.close(fig)

    elif chart == "Scatter Plot":
        x_col = st.selectbox("X axis", num_cols, index=0)
        y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1))
        color_col = st.selectbox("Colour by", ["None"] + cat_cols)
        trend = st.checkbox("Linear trend line", value=True)

        fig, ax = nice_fig((7, 5))
        if color_col != "None":
            groups = df[color_col].dropna().unique()
            cmap = plt.cm.Set1(np.linspace(0, 1, len(groups)))
            for grp, clr in zip(groups, cmap):
                mask = df[color_col] == grp
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                           label=str(grp), color=clr, alpha=0.7, s=40)
            ax.legend(title=color_col)
        else:
            ax.scatter(df[x_col], df[y_col], color="#4c9be8", alpha=0.65, s=40)
        if trend:
            valid = df[[x_col, y_col]].dropna()
            m, b, r, p, _ = stats.linregress(valid[x_col], valid[y_col])
            xs = np.linspace(valid[x_col].min(), valid[x_col].max(), 200)
            ax.plot(xs, m * xs + b, "r--", lw=1.5, label=f"y={m:.2f}x+{b:.2f}  r={r:.2f}")
            ax.legend()
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"Scatter — {x_col} vs {y_col}")
        st.pyplot(fig, width="content"); plt.close(fig)
        r, p = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna())
        st.caption(f"Pearson r = **{r:.4f}**, p = **{p:.4f}** {p_stars(p)}")

    elif chart == "Bar Chart":
        if not cat_cols:
            st.info("No categorical columns available.")
        else:
            cat = st.selectbox("Category column", cat_cols)
            val = st.selectbox("Value column (mean)", num_cols)
            agg = df.groupby(cat)[val].mean().sort_values(ascending=False)
            fig, ax = nice_fig((max(5, len(agg) * 0.8), 4))
            cmap = plt.cm.Set2(np.linspace(0, 1, len(agg)))
            ax.bar(agg.index.astype(str), agg.values, color=cmap, edgecolor="white")
            ax.set_xlabel(cat); ax.set_ylabel(f"Mean {val}")
            ax.set_title(f"Mean {val} by {cat}")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig, width="content"); plt.close(fig)

    elif chart == "Line Chart":
        y_cols = st.multiselect("Y columns", num_cols, default=num_cols[:min(2, len(num_cols))])
        x_opt = st.selectbox("X axis", ["Row Index"] + num_cols + cat_cols)
        if y_cols:
            fig, ax = nice_fig((9, 4))
            x_vals = df.index if x_opt == "Row Index" else df[x_opt]
            for yc in y_cols:
                ax.plot(x_vals, df[yc], label=yc, lw=1.5)
            ax.set_xlabel(x_opt); ax.legend(); ax.set_title("Line Chart")
            st.pyplot(fig, width="content"); plt.close(fig)

    elif chart == "Q-Q Plot":
        col = st.selectbox("Column", num_cols)
        data = df[col].dropna()
        fig, ax = nice_fig((5, 5))
        (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, color="#4c9be8", s=20, alpha=0.7, label="Data")
        ax.plot([min(osm), max(osm)],
                [slope * min(osm) + intercept, slope * max(osm) + intercept],
                "r--", lw=2, label="Normal line")
        ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"Q-Q Plot — {col}"); ax.legend()
        st.pyplot(fig, width="content"); plt.close(fig)
        st.caption("Points close to the red line → data is approximately normal.")

    elif chart == "Correlation Heatmap":
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            corr = df[num_cols].corr()
            n = len(num_cols)
            fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(n)); ax.set_xticklabels(num_cols, rotation=45, ha="right")
            ax.set_yticks(range(n)); ax.set_yticklabels(num_cols)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if abs(corr.values[i,j]) > 0.6 else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Pearson Correlation Heatmap")
            plt.tight_layout()
            st.pyplot(fig, width="content"); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Probability Distributions":
    st.header("Probability Distributions",
              help="Explore PDF/PMF and CDF of common distributions, or fit distributions "
                   "to your data with goodness-of-fit tests.")
    tab_explore, tab_fit = st.tabs(["Distribution Explorer", "Fit to Data"])

    with tab_explore:
        dist_name = st.selectbox("Distribution", [
            "Normal", "Binomial", "Poisson", "Bernoulli", "Uniform", "Exponential",
        ])
        show_pdf = st.checkbox("Show PDF / PMF", value=True)
        show_cdf = st.checkbox("Show CDF", value=True)

        n_plots = sum([show_pdf, show_cdf])
        if n_plots == 0:
            st.info("Select at least one of PDF or CDF.")
            st.stop()

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        _idx = [0]

        def get_ax():
            a = axes[_idx[0]]; _idx[0] += 1
            a.spines[["top", "right"]].set_visible(False)
            return a

        if dist_name == "Normal":
            mu = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.5)
            sigma = st.slider("Std Dev (σ)", 0.1, 10.0, 1.0, 0.1)
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
            if show_pdf:
                ax = get_ax()
                ax.plot(x, norm.pdf(x, mu, sigma), "#4c9be8", lw=2)
                ax.fill_between(x, norm.pdf(x, mu, sigma), alpha=0.2, color="#4c9be8")
                ax.set_title(f"Normal PDF  μ={mu}, σ={sigma}")
                ax.set_xlabel("x"); ax.set_ylabel("f(x)")
            if show_cdf:
                ax = get_ax()
                ax.plot(x, norm.cdf(x, mu, sigma), "crimson", lw=2)
                ax.set_title(f"Normal CDF  μ={mu}, σ={sigma}")
                ax.set_xlabel("x"); ax.set_ylabel("F(x)"); ax.set_ylim(0, 1)
            st.markdown(f"**E[X] = μ = {mu}** &nbsp;|&nbsp; **Var[X] = σ² = {sigma**2:.3f}**")

        elif dist_name == "Binomial":
            n = st.slider("n (trials)", 1, 100, 20)
            p = st.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01)
            k = np.arange(0, n + 1)
            if show_pdf:
                ax = get_ax()
                ax.bar(k, binom.pmf(k, n, p), color="#4c9be8", edgecolor="white")
                ax.set_title(f"Binomial PMF  n={n}, p={p}")
                ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
            if show_cdf:
                ax = get_ax()
                ax.step(k, binom.cdf(k, n, p), "crimson", lw=2, where="post")
                ax.set_title(f"Binomial CDF  n={n}, p={p}")
                ax.set_xlabel("k"); ax.set_ylabel("P(X≤k)"); ax.set_ylim(0, 1)
            st.markdown(f"**E[X] = np = {n*p:.2f}** &nbsp;|&nbsp; **Var[X] = np(1−p) = {n*p*(1-p):.4f}**")

        elif dist_name == "Poisson":
            lam = st.slider("λ (rate)", 0.1, 30.0, 5.0, 0.1)
            k = np.arange(0, int(lam * 3) + 1)
            if show_pdf:
                ax = get_ax()
                ax.bar(k, poisson.pmf(k, lam), color="#4c9be8", edgecolor="white")
                ax.set_title(f"Poisson PMF  λ={lam}")
                ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
            if show_cdf:
                ax = get_ax()
                ax.step(k, poisson.cdf(k, lam), "crimson", lw=2, where="post")
                ax.set_title(f"Poisson CDF  λ={lam}")
                ax.set_xlabel("k"); ax.set_ylabel("P(X≤k)"); ax.set_ylim(0, 1)
            st.markdown(f"**E[X] = λ = {lam}** &nbsp;|&nbsp; **Var[X] = λ = {lam}**")

        elif dist_name == "Bernoulli":
            p = st.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01)
            k = np.array([0, 1])
            if show_pdf:
                ax = get_ax()
                ax.bar(["0 (Failure)", "1 (Success)"], bernoulli.pmf(k, p),
                       color=["#e87c7c", "#4c9be8"], edgecolor="white")
                ax.set_title(f"Bernoulli PMF  p={p}"); ax.set_ylabel("P(X=k)")
            if show_cdf:
                ax = get_ax()
                ax.bar(["0", "1"], bernoulli.cdf(k, p),
                       color=["#e87c7c", "#4c9be8"], edgecolor="white")
                ax.set_title(f"Bernoulli CDF  p={p}")
                ax.set_ylabel("P(X≤k)"); ax.set_ylim(0, 1.15)
            st.markdown(f"**E[X] = p = {p}** &nbsp;|&nbsp; **Var[X] = p(1−p) = {p*(1-p):.4f}**")

        elif dist_name == "Uniform":
            a_u = st.slider("a (min)", -20.0, 19.0, 0.0, 0.5)
            b_u = st.slider("b (max)", a_u + 0.1, 20.0, float(a_u + 5), 0.5)
            x = np.linspace(a_u - 1, b_u + 1, 500)
            if show_pdf:
                ax = get_ax()
                ax.plot(x, sp_uniform.pdf(x, a_u, b_u - a_u), "#4c9be8", lw=2)
                ax.fill_between(x, sp_uniform.pdf(x, a_u, b_u - a_u), alpha=0.2, color="#4c9be8")
                ax.set_title(f"Uniform PDF  [{a_u}, {b_u}]")
                ax.set_xlabel("x"); ax.set_ylabel("f(x)")
            if show_cdf:
                ax = get_ax()
                ax.plot(x, sp_uniform.cdf(x, a_u, b_u - a_u), "crimson", lw=2)
                ax.set_title(f"Uniform CDF  [{a_u}, {b_u}]")
                ax.set_xlabel("x"); ax.set_ylabel("F(x)"); ax.set_ylim(-0.05, 1.05)
            st.markdown(f"**E[X] = (a+b)/2 = {(a_u+b_u)/2:.2f}** &nbsp;|&nbsp; **Var[X] = (b−a)²/12 = {(b_u-a_u)**2/12:.4f}**")

        elif dist_name == "Exponential":
            lam_e = st.slider("λ (rate)", 0.05, 5.0, 1.0, 0.05)
            scale = 1 / lam_e
            x = np.linspace(0, 8 / lam_e, 400)
            if show_pdf:
                ax = get_ax()
                ax.plot(x, expon.pdf(x, scale=scale), "#4c9be8", lw=2)
                ax.fill_between(x, expon.pdf(x, scale=scale), alpha=0.2, color="#4c9be8")
                ax.set_title(f"Exponential PDF  λ={lam_e}")
                ax.set_xlabel("x"); ax.set_ylabel("f(x)")
            if show_cdf:
                ax = get_ax()
                ax.plot(x, expon.cdf(x, scale=scale), "crimson", lw=2)
                ax.set_title(f"Exponential CDF  λ={lam_e}")
                ax.set_xlabel("x"); ax.set_ylabel("F(x)"); ax.set_ylim(0, 1.05)
            st.markdown(f"**E[X] = 1/λ = {scale:.4f}** &nbsp;|&nbsp; **Var[X] = 1/λ² = {scale**2:.4f}**")

        plt.tight_layout()
        st.pyplot(fig, width="content")
        plt.close(fig)

    with tab_fit:
        st.subheader("Distribution Fitting & Goodness-of-Fit Tests")
        st.markdown("Fit multiple distributions and perform comprehensive goodness-of-fit tests.")
        
        df_fit = st.session_state.get("df")
        if df_fit is None:
            st.info("Load a dataset in the Data Explorer first.")
        else:
            num_fit = df_fit.select_dtypes(include=np.number).columns.tolist()
            col_fit = st.selectbox("Select column", num_fit, key="fit_col")
            data_fit = df_fit[col_fit].dropna()

            if len(data_fit) < 10:
                st.error("Need at least 10 observations for distribution fitting.")
                st.stop()

            st.markdown("---")
            test_type = st.radio("Test Type", ["Automatic Fitting (Multiple Distributions)", "Chi-Square Goodness-of-Fit", "Anderson-Darling Test"], horizontal=False)

            if test_type == "Automatic Fitting (Multiple Distributions)":
                st.markdown("**Fits 6 distributions and compares using K-S test.**")
                
                # Candidate distributions and their fitted parameters
                fit_results = []
                data_pos = data_fit[data_fit > 0]  # for distributions requiring positive values

                candidates = {
                    "Normal":      ("norm",    norm,       norm.fit(data_fit)),
                    "Log-Normal":  ("lognorm", lognorm,    lognorm.fit(data_pos, floc=0) if len(data_pos) > 10 else None),
                    "Exponential": ("expon",   expon,      expon.fit(data_pos, floc=0) if len(data_pos) > 10 else None),
                    "Gamma":       ("gamma",   gamma,      gamma.fit(data_pos, floc=0) if len(data_pos) > 10 else None),
                    "Uniform":     ("uniform", sp_uniform, sp_uniform.fit(data_fit)),
                    "Beta":        ("beta",    beta,       beta.fit((data_fit - data_fit.min()) / (data_fit.max() - data_fit.min() + 1e-9)) if data_fit.max() > data_fit.min() else None),
                }

                xs_fit = np.linspace(float(data_fit.min()), float(data_fit.max()), 300)
                fig, ax = nice_fig((10, 5))
                ax.hist(data_fit, bins=30, density=True, color="#4c9be8",
                        edgecolor="white", alpha=0.6, label="Data")

                colors_fit = ["crimson", "green", "orange", "purple", "brown", "pink"]
                for (name, (scipy_name, dist_obj, params)), color in zip(candidates.items(), colors_fit):
                    if params is None:
                        continue
                    try:
                        # K-S test
                        if name == "Beta":
                            # For beta, use transformed data
                            data_transformed = (data_fit - data_fit.min()) / (data_fit.max() - data_fit.min() + 1e-9)
                            ks_s, ks_p = kstest(data_transformed, scipy_name, args=params)
                            # Plot on original scale
                            xs_beta = (xs_fit - data_fit.min()) / (data_fit.max() - data_fit.min() + 1e-9)
                            y_fit = dist_obj.pdf(xs_beta, *params) / (data_fit.max() - data_fit.min() + 1e-9)
                        else:
                            ks_s, ks_p = kstest(data_fit if name in ["Normal", "Uniform"] else data_pos, 
                                               scipy_name, args=params)
                            y_fit = dist_obj.pdf(xs_fit, *params)
                        
                        ax.plot(xs_fit, y_fit, color=color, lw=2.5, label=f"{name} (K-S p={ks_p:.3f})", alpha=0.8)
                        
                        # Parameter strings
                        if name == "Normal":
                            param_str = f"μ={params[0]:.2f}, σ={params[1]:.2f}"
                        elif name == "Log-Normal":
                            param_str = f"s={params[0]:.2f}, scale={params[2]:.2f}"
                        elif name == "Exponential":
                            param_str = f"λ={1/params[1]:.4f}"
                        elif name == "Gamma":
                            param_str = f"α={params[0]:.2f}, β={params[2]:.2f}"
                        elif name == "Uniform":
                            param_str = f"[{params[0]:.2f}, {params[0]+params[1]:.2f}]"
                        elif name == "Beta":
                            param_str = f"α={params[0]:.2f}, β={params[1]:.2f}"
                        else:
                            param_str = str(params)
                            
                        fit_results.append({
                            "Distribution": name,
                            "Parameters": param_str,
                            "K-S Statistic": round(ks_s, 4),
                            "K-S p-value": round(ks_p, 4),
                            "Fits Well?": ":material/check_circle: Yes" if ks_p > 0.05 else ":material/cancel: No",
                        })
                    except Exception as e:
                        pass

                ax.set_xlabel(col_fit); ax.set_ylabel("Density")
                ax.set_title(f"Distribution Fitting — {col_fit}")
                ax.legend(fontsize=9)
                st.pyplot(fig, width="content")
                plt.close(fig)

                if fit_results:
                    fit_df = pd.DataFrame(fit_results).sort_values("K-S p-value", ascending=False)
                    st.dataframe(fit_df, width="stretch", hide_index=True)
                    best = fit_df.iloc[0]["Distribution"]
                    best_p = fit_df.iloc[0]["K-S p-value"]
                    st.success(f"**Best fitting distribution:** {best} (K-S p-value = {best_p:.4f})")
                    st.caption("**Interpretation:** Higher p-value = better fit. p > 0.05 suggests good fit.")

            elif test_type == "Chi-Square Goodness-of-Fit":
                st.markdown("**Chi-Square test:** Compares observed frequencies to expected frequencies from a theoretical distribution.")
                
                dist_choice = st.selectbox("Theoretical Distribution", ["Normal", "Uniform", "Exponential", "Custom Expected Frequencies"])
                n_bins = st.slider("Number of bins", 5, 30, 10)

                if st.button("Run Chi-Square Test"):
                    # Create histogram bins
                    observed_freq, bin_edges = np.histogram(data_fit, bins=n_bins)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_width = bin_edges[1] - bin_edges[0]

                    if dist_choice == "Normal":
                        # Fit normal distribution
                        mu, sigma = norm.fit(data_fit)
                        # Expected frequencies
                        expected_probs = np.diff(norm.cdf(bin_edges, mu, sigma))
                        expected_freq = expected_probs * len(data_fit)
                        dist_params = f"μ={mu:.2f}, σ={sigma:.2f}"
                        
                    elif dist_choice == "Uniform":
                        # Uniform distribution
                        expected_freq = np.ones(n_bins) * len(data_fit) / n_bins
                        dist_params = f"[{data_fit.min():.2f}, {data_fit.max():.2f}]"
                        
                    elif dist_choice == "Exponential":
                        data_pos = data_fit[data_fit > 0]
                        if len(data_pos) < 10:
                            st.error("Need positive values for exponential distribution.")
                            st.stop()
                        # Fit exponential
                        loc, scale = expon.fit(data_pos, floc=0)
                        expected_probs = np.diff(expon.cdf(bin_edges, loc, scale))
                        expected_freq = expected_probs * len(data_fit)
                        dist_params = f"λ={1/scale:.4f}"
                    
                    # Combine bins with expected frequency < 5 (chi-square assumption)
                    combined_obs = []
                    combined_exp = []
                    temp_obs = 0
                    temp_exp = 0
                    
                    for obs, exp in zip(observed_freq, expected_freq):
                        temp_obs += obs
                        temp_exp += exp
                        if temp_exp >= 5:
                            combined_obs.append(temp_obs)
                            combined_exp.append(temp_exp)
                            temp_obs = 0
                            temp_exp = 0
                    
                    # Add remainder
                    if temp_obs > 0:
                        if len(combined_obs) > 0:
                            combined_obs[-1] += temp_obs
                            combined_exp[-1] += temp_exp
                        else:
                            combined_obs.append(temp_obs)
                            combined_exp.append(temp_exp)
                    
                    combined_obs = np.array(combined_obs)
                    combined_exp = np.array(combined_exp)
                    
                    # Chi-square test
                    chi2_stat = np.sum((combined_obs - combined_exp)**2 / combined_exp)
                    df_chi = len(combined_obs) - 1 - (2 if dist_choice in ["Normal", "Exponential"] else 0)  # subtract estimated parameters
                    if df_chi <= 0:
                        st.error("Not enough degrees of freedom. Try fewer bins or more data.")
                        st.stop()
                    
                    p_value = 1 - chi2.cdf(chi2_stat, df_chi)
                    
                    st.write(f"**Distribution:** {dist_choice} ({dist_params})")
                    st.write(f"**Number of bins:** {n_bins} (combined to {len(combined_obs)} bins with expected ≥ 5)")
                    st.write(f"**Chi-Square Statistic:** {chi2_stat:.4f}")
                    st.write(f"**Degrees of Freedom:** {df_chi}")
                    st.write(f"**p-value:** {p_value:.4f} {p_stars(p_value)}")
                    
                    if p_value > 0.05:
                        st.success(f":material/check_circle: **Fail to reject H₀** (p = {p_value:.4f} > 0.05). Data fits {dist_choice} distribution well.")
                    else:
                        st.warning(f":material/cancel: **Reject H₀** (p = {p_value:.4f} ≤ 0.05). Data does NOT fit {dist_choice} distribution.")
                    
                    # Visualization
                    fig, ax = nice_fig((10, 5))
                    x_pos = np.arange(len(combined_obs))
                    width = 0.35
                    
                    ax.bar(x_pos - width/2, combined_obs, width, label='Observed', color='#4c9be8', alpha=0.7)
                    ax.bar(x_pos + width/2, combined_exp, width, label='Expected', color='#e89b4c', alpha=0.7)
                    
                    ax.set_xlabel('Bin Group')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Chi-Square Goodness-of-Fit: {dist_choice} Distribution')
                    ax.legend()
                    ax.spines[["top", "right"]].set_visible(False)
                    
                    st.pyplot(fig, width="content")
                    plt.close(fig)
                    
                    st.caption("**Chi-Square Test Assumptions:** Expected frequency ≥ 5 in each bin (bins combined automatically).")

            elif test_type == "Anderson-Darling Test":
                st.markdown("**Anderson-Darling test:** More sensitive to deviations in the tails than K-S test. Tests if data comes from a specific distribution.")
                
                dist_choice = st.selectbox("Distribution to test", ["Normal", "Exponential", "Logistic", "Gumbel"])
                
                if st.button("Run Anderson-Darling Test"):
                    if dist_choice == "Normal":
                        result = anderson(data_fit, dist='norm')
                        dist_name = "Normal"
                    elif dist_choice == "Exponential":
                        result = anderson(data_fit, dist='expon')
                        dist_name = "Exponential"
                    elif dist_choice == "Logistic":
                        result = anderson(data_fit, dist='logistic')
                        dist_name = "Logistic"
                    elif dist_choice == "Gumbel":
                        result = anderson(data_fit, dist='gumbel')
                        dist_name = "Gumbel (Extreme Value)"
                    
                    st.write(f"**Testing distribution:** {dist_name}")
                    st.write(f"**Anderson-Darling Statistic:** {result.statistic:.4f}")
                    st.write(f"**Sample Size:** {len(data_fit)}")
                    
                    st.markdown("**Critical Values and Significance Levels:**")
                    results_table = pd.DataFrame({
                        "Significance Level": [f"{sl}%" for sl in result.significance_level],
                        "Critical Value": result.critical_values,
                        "Result": [":material/check_circle: Fail to reject H₀" if result.statistic < cv else ":material/cancel: Reject H₀" 
                                  for cv in result.critical_values]
                    })
                    st.dataframe(results_table, width="stretch", hide_index=True)
                    
                    # Determine overall conclusion
                    if result.statistic < result.critical_values[2]:  # 5% level (index 2)
                        st.success(f":material/check_circle: **Data appears to follow {dist_name} distribution** (at 5% significance level)")
                    else:
                        st.warning(f":material/cancel: **Data does NOT follow {dist_name} distribution** (at 5% significance level)")
                    
                    st.caption("**Interpretation:** If test statistic < critical value, fail to reject H₀ (data fits distribution).")


# ══════════════════════════════════════════════════════════════════════════════
# CLT DEMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "CLT Demonstration":
    st.header("Central Limit Theorem (CLT) Demonstration",
              help="Simulate sampling distributions of the mean for different sample sizes "
                   "to see how they converge to a Normal distribution.")
    st.markdown("""
**The Central Limit Theorem (CLT) states** that the distribution of sample means converges to a Normal distribution 
as sample size increases - regardless of the original distribution's shape.

$$E(\\bar{X}) = \\mu \\qquad SE(\\bar{X}) = \\frac{\\sigma}{\\sqrt{n}}$$
$$Z = \\frac{\\bar{X} - \\mu}{\\sigma / \\sqrt{n}}$$
""")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found for CLT analysis.")
        st.stop()

    col_clt = st.selectbox("Select column for CLT demonstration", num_cols)
    data_clt = df[col_clt].dropna().values

    if len(data_clt) < 10:
        st.warning("Need at least 10 data points for meaningful CLT demonstration.")
        st.stop()

    # Settings
    n_simulations = st.slider("Number of simulations", 100, 5000, 1000, 100)
    sample_sizes = st.multiselect("Sample sizes to compare", [5, 10, 20, 30, 50, 100],
                                  default=[5, 30])

    if len(sample_sizes) == 0:
        st.info("Select at least one sample size.")
        st.stop()

    # Population parameters
    pop_mean = np.mean(data_clt)
    pop_std = np.std(data_clt, ddof=1)

    st.subheader("Population Distribution")
    col1, col2, col3 = st.columns(3)
    col1.metric("Population Mean (μ)", f"{pop_mean:.4f}")
    col2.metric("Population Std Dev (σ)", f"{pop_std:.4f}")
    col3.metric("Sample Size (N)", len(data_clt))

    # Original distribution
    fig_pop, ax_pop = nice_fig((10, 4))
    ax_pop.hist(data_clt, bins=30, alpha=0.7, color="#4c9be8", edgecolor="white", density=True)
    ax_pop.axvline(pop_mean, color='red', linestyle='--', lw=2, label=f'Mean = {pop_mean:.2f}')
    ax_pop.set_xlabel(col_clt)
    ax_pop.set_ylabel("Density")
    ax_pop.set_title(f"Original Data Distribution — {col_clt}")
    ax_pop.legend()
    st.pyplot(fig_pop, width="content")
    plt.close(fig_pop)

    # CLT Demonstration for different sample sizes
    st.subheader("Sampling Distribution of Sample Means (CLT Effect)")

    n_plots = len(sample_sizes)
    fig_clt, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for idx, n in enumerate(sample_sizes):
        # Generate sampling distribution
        sample_means = []
        for _ in range(n_simulations):
            sample = np.random.choice(data_clt, size=min(n, len(data_clt)), replace=True)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        # Theoretical SE
        se_theoretical = pop_std / np.sqrt(n)

        # Plot
        ax = axes[idx]
        ax.hist(sample_means, bins=40, alpha=0.7, color="#4c9be8", edgecolor="white",
                density=True, label="Sampling Distribution")

        # Overlay theoretical normal
        x_range = np.linspace(sample_means.min(), sample_means.max(), 200)
        ax.plot(x_range, norm.pdf(x_range, pop_mean, se_theoretical),
                'r-', lw=2, label=f'Normal(μ={pop_mean:.2f}, SE={se_theoretical:.4f})')

        ax.axvline(pop_mean, color='darkred', linestyle='--', lw=1.5, alpha=0.7)
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        # Stats
        actual_mean = np.mean(sample_means)
        actual_se = np.std(sample_means, ddof=1)

        ax.text(0.05, 0.95, f"Actual Mean: {actual_mean:.4f}\nActual SE: {actual_se:.4f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_clt, width="content")
    plt.close(fig_clt)

    st.info("**Key Observation:** As sample size increases (n:material/arrow_upward:), the sampling distribution becomes more " +
            "normal and the standard error decreases (SE = σ/√n). This demonstrates the Central Limit Theorem!")

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Statistical Inference":
    st.header("Statistical Inference — Hypothesis Testing",
              help="Run t-tests (one/two-sample, paired), ANOVA, chi-square, correlation "
                   "and normality tests with adjustable significance level.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if not num_cols:
        st.error("No numeric columns found for statistical tests.")
        st.stop()

    # Test selection
    test_type = st.selectbox("Select Test Type", [
        "One-Sample t-Test",
        "Two-Sample t-Test (Independent)",
        "Paired t-Test",
        "One-Way ANOVA",
        "Chi-Square Test",
        "Correlation Test",
        "Normality Tests",
    ])

    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)

    st.markdown("---")

    # ─── ONE-SAMPLE T-TEST ───
    if test_type == "One-Sample t-Test":
        st.subheader("One-Sample t-Test")
        st.markdown("Test if the sample mean differs from a hypothesized population mean.")

        col = st.selectbox("Select column", num_cols)
        mu0 = st.number_input("Hypothesized mean (μ₀)", value=0.0, step=0.1)
        tail = st.radio("Alternative hypothesis", ["Two-sided", "Less than", "Greater than"])

        if st.button("Run Test"):
            data = df[col].dropna()

            if len(data) < 2:
                st.error("Need at least 2 observations.")
            else:
                alt_map = {"Two-sided": "two-sided", "Less than": "less", "Greater than": "greater"}
                t_stat, p_val = ttest_1samp(data, mu0, alternative=alt_map[tail])

                st.write(f"**Sample Mean:** {data.mean():.4f}")
                st.write(f"**Sample Std Dev:** {data.std(ddof=1):.4f}")
                st.write(f"**Sample Size:** {len(data)}")
                st.write(f"**t-statistic:** {t_stat:.4f}")
                st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                if p_val < alpha:
                    st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                              f"The sample mean significantly differs from {mu0}.")
                else:
                    st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                           f"No significant difference from {mu0}.")

    # ─── TWO-SAMPLE T-TEST ───
    elif test_type == "Two-Sample t-Test (Independent)":
        st.subheader("Two-Sample t-Test (Independent Groups)")
        st.markdown("Compare means of two independent groups.")

        col = st.selectbox("Select numeric column", num_cols)
        if not cat_cols:
            st.error("Need at least one categorical column to define groups.")
            st.stop()

        grp_col = st.selectbox("Select grouping column", cat_cols)
        equal_var = st.checkbox("Assume equal variances", value=False)

        if st.button("Run Test"):
            groups = df[grp_col].dropna().unique()
            if len(groups) < 2:
                st.error("Need at least 2 groups.")
            else:
                g1, g2 = groups[:2]
                data1 = df[df[grp_col] == g1][col].dropna()
                data2 = df[df[grp_col] == g2][col].dropna()

                if len(data1) < 2 or len(data2) < 2:
                    st.error("Each group needs at least 2 observations.")
                else:
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=equal_var)

                    st.write(f"**Group 1 ({g1}):** Mean = {data1.mean():.4f}, SD = {data1.std(ddof=1):.4f}, n = {len(data1)}")
                    st.write(f"**Group 2 ({g2}):** Mean = {data2.mean():.4f}, SD = {data2.std(ddof=1):.4f}, n = {len(data2)}")
                    st.write(f"**t-statistic:** {t_stat:.4f}")
                    st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                    if p_val < alpha:
                        st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                                  "Significant difference between groups.")
                    else:
                        st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                               "No significant difference.")

                    # Levene's test for variance equality
                    lev_stat, lev_p = levene(data1, data2)
                    st.caption(f"**Levene's Test for Equal Variances:** p = {lev_p:.4f} " +
                              ("(variances are equal)" if lev_p > 0.05 else "(variances differ)"))

    # ─── PAIRED T-TEST ───
    elif test_type == "Paired t-Test":
        st.subheader("Paired t-Test")
        st.markdown("Compare two related samples (e.g., before/after measurements).")

        col1 = st.selectbox("First measurement", num_cols, key="pair1")
        col2 = st.selectbox("Second measurement", num_cols, key="pair2")

        if st.button("Run Test"):
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()

            # Use only complete pairs
            valid_idx = df[[col1, col2]].dropna().index
            data1 = df.loc[valid_idx, col1]
            data2 = df.loc[valid_idx, col2]

            if len(data1) < 2:
                st.error("Need at least 2 paired observations.")
            else:
                t_stat, p_val = ttest_rel(data1, data2)
                diff = data1 - data2

                st.write(f"**Mean of differences:** {diff.mean():.4f}")
                st.write(f"**SD of differences:** {diff.std(ddof=1):.4f}")
                st.write(f"**Number of pairs:** {len(data1)}")
                st.write(f"**t-statistic:** {t_stat:.4f}")
                st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                if p_val < alpha:
                    st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                              "Significant difference between paired measurements.")
                else:
                    st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                           "No significant difference.")

    # ─── ONE-WAY ANOVA ───
    elif test_type == "One-Way ANOVA":
        st.subheader("One-Way ANOVA")
        st.markdown("Compare means across 3+ independent groups.")

        col = st.selectbox("Select numeric column", num_cols)
        if not cat_cols:
            st.error("Need a categorical column to define groups.")
            st.stop()

        grp_col = st.selectbox("Select grouping column", cat_cols)

        if st.button("Run Test"):
            groups = df[grp_col].dropna().unique()
            if len(groups) < 2:
                st.error("Need at least 2 groups for ANOVA.")
            else:
                group_data = [df[df[grp_col] == g][col].dropna() for g in groups]
                group_data = [g for g in group_data if len(g) > 0]

                if len(group_data) < 2:
                    st.error("Need at least 2 non-empty groups.")
                else:
                    f_stat, p_val = f_oneway(*group_data)

                    st.write(f"**F-statistic:** {f_stat:.4f}")
                    st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                    # Group summaries
                    for idx, g in enumerate(groups[:len(group_data)]):
                        data_g = group_data[idx]
                        st.write(f"**{g}:** Mean = {data_g.mean():.4f}, SD = {data_g.std(ddof=1):.4f}, n = {len(data_g)}")

                    if p_val < alpha:
                        st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                                  "At least one group mean differs significantly.")

                        # Tukey HSD post-hoc
                        try:
                            df_anova = df[[col, grp_col]].dropna()
                            tukey = pairwise_tukeyhsd(df_anova[col], df_anova[grp_col], alpha=alpha)
                            st.write("**Tukey HSD Post-hoc Test:**")
                            st.text(str(tukey))
                        except:
                            st.warning("Could not perform Tukey HSD post-hoc test.")
                    else:
                        st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                               "No significant differences among group means.")

    # ─── CHI-SQUARE TEST ───
    elif test_type == "Chi-Square Test":
        st.subheader("Chi-Square Test of Independence")
        st.markdown("Test association between two categorical variables.")

        if len(cat_cols) < 2:
            st.error("Need at least 2 categorical columns.")
            st.stop()

        col1 = st.selectbox("First categorical variable", cat_cols, key="chi1")
        col2 = st.selectbox("Second categorical variable", cat_cols, key="chi2")

        if st.button("Run Test"):
            contingency_table = pd.crosstab(df[col1], df[col2])

            if contingency_table.size < 4:
                st.error("Need at least 2x2 contingency table.")
            else:
                chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

                st.write("**Contingency Table (Observed):**")
                st.dataframe(contingency_table)

                st.write(f"**χ² statistic:** {chi2_stat:.4f}")
                st.write(f"**Degrees of freedom:** {dof}")
                st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                if p_val < alpha:
                    st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                              "Variables are associated.")
                else:
                    st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                           "Variables are independent.")

    # ─── CORRELATION TEST ───
    elif test_type == "Correlation Test":
        st.subheader("Correlation Test (Pearson)")
        st.markdown("Test if two numeric variables are linearly correlated.")

        col1 = st.selectbox("First variable", num_cols, key="corr1")
        col2 = st.selectbox("Second variable", num_cols, key="corr2")

        if st.button("Run Test"):
            data_corr = df[[col1, col2]].dropna()

            if len(data_corr) < 3:
                st.error("Need at least 3 paired observations.")
            else:
                r, p_val = stats.pearsonr(data_corr[col1], data_corr[col2])

                st.write(f"**Pearson correlation (r):** {r:.4f}")
                st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")
                st.write(f"**Sample size:** {len(data_corr)}")

                if p_val < alpha:
                    st.success(f":material/check_circle: **Significant correlation** (p = {p_val:.4f} < α = {alpha}). " +
                              f"r = {r:.4f}")
                else:
                    st.info(f":material/cancel: **No significant correlation** (p = {p_val:.4f} ≥ α = {alpha}).")

                # Scatter plot
                fig, ax = nice_fig((7, 5))
                ax.scatter(data_corr[col1], data_corr[col2], alpha=0.6, color="#4c9be8")
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"Correlation: r = {r:.4f}, p = {p_val:.4f}")
                st.pyplot(fig, width="content")
                plt.close(fig)

    # ─── NORMALITY TESTS ───
    elif test_type == "Normality Tests":
        st.subheader("Normality Tests")
        st.markdown("Test if data follows a normal distribution.")

        col = st.selectbox("Select column", num_cols)

        if st.button("Run Tests"):
            data = df[col].dropna()

            if len(data) < 3:
                st.error("Need at least 3 observations.")
            else:
                # Generate normality table
                norm_df = normality_table(data)
                st.dataframe(norm_df, width="stretch", hide_index=True)

                # Q-Q plot
                fig, ax = nice_fig((6, 5))
                (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
                ax.scatter(osm, osr, color="#4c9be8", s=20, alpha=0.7, label="Data")
                ax.plot([min(osm), max(osm)],
                        [slope * min(osm) + intercept, slope * max(osm) + intercept],
                        "r--", lw=2, label="Normal line")
                ax.set_xlabel("Theoretical Quantiles")
                ax.set_ylabel("Sample Quantiles")
                ax.set_title(f"Q-Q Plot — {col}")
                ax.legend()
                st.pyplot(fig, width="content")
                plt.close(fig)

                st.caption("**Interpretation:** If p > 0.05 in any test, data may be approximately normal. " +
                          "Q-Q plot points close to the line suggest normality.")

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Regression Analysis":
    st.header("Regression Analysis",
              help="Fit simple or multiple linear regression models, view coefficients, "
                   "R-squared, and diagnostic plots.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns for regression.")
        st.stop()

    reg_type = st.selectbox("Regression Type", ["Simple Linear Regression", "Multiple Linear Regression"])

    if reg_type == "Simple Linear Regression":
        st.subheader("Simple Linear Regression")
        st.markdown("Fit a linear model: **y = β₀ + β₁x + ε**")

        x_col = st.selectbox("Independent variable (X)", num_cols, key="reg_x")
        y_col = st.selectbox("Dependent variable (Y)", num_cols, key="reg_y")

        if st.button("Fit Model"):
            data_reg = df[[x_col, y_col]].dropna()

            if len(data_reg) < 3:
                st.error("Need at least 3 observations.")
            else:
                X = data_reg[x_col].values.reshape(-1, 1)
                y = data_reg[y_col].values

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                # Model parameters
                beta0 = model.intercept_
                beta1 = model.coef_[0]
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)

                # Adjusted R²
                n = len(data_reg)
                p = 1  # one predictor
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

                # Display results
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Intercept (β₀)", f"{beta0:.4f}")
                col2.metric("Slope (β₁)", f"{beta1:.4f}")
                col3.metric("R²", f"{r2:.4f}")
                col4.metric("RMSE", f"{rmse:.4f}")

                st.write(f"**Regression Equation:** {y_col} = {beta0:.4f} + {beta1:.4f} × {x_col}")
                st.write(f"**Adjusted R²:** {adj_r2:.4f}")

                # Residuals
                residuals = y - y_pred

                # Plots
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # 1. Regression line
                axes[0, 0].scatter(X, y, alpha=0.6, color="#4c9be8", s=30, label="Data")
                axes[0, 0].plot(X, y_pred, "r-", lw=2, label="Fitted line")
                axes[0, 0].set_xlabel(x_col)
                axes[0, 0].set_ylabel(y_col)
                axes[0, 0].set_title(f"Regression Line (R² = {r2:.4f})")
                axes[0, 0].legend()
                axes[0, 0].spines[["top", "right"]].set_visible(False)

                # 2. Residuals vs Fitted
                axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color="#4c9be8", s=30)
                axes[0, 1].axhline(0, color="red", linestyle="--", lw=2)
                axes[0, 1].set_xlabel("Fitted values")
                axes[0, 1].set_ylabel("Residuals")
                axes[0, 1].set_title("Residuals vs Fitted")
                axes[0, 1].spines[["top", "right"]].set_visible(False)

                # 3. Q-Q plot of residuals
                (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
                axes[1, 0].scatter(osm, osr, color="#4c9be8", s=20, alpha=0.7)
                axes[1, 0].plot([min(osm), max(osm)],
                                [slope * min(osm) + intercept, slope * max(osm) + intercept],
                                "r--", lw=2)
                axes[1, 0].set_xlabel("Theoretical Quantiles")
                axes[1, 0].set_ylabel("Sample Quantiles")
                axes[1, 0].set_title("Q-Q Plot (Residuals)")
                axes[1, 0].spines[["top", "right"]].set_visible(False)

                # 4. Histogram of residuals
                axes[1, 1].hist(residuals, bins=20, alpha=0.7, color="#4c9be8", edgecolor="white")
                axes[1, 1].axvline(0, color="red", linestyle="--", lw=2)
                axes[1, 1].set_xlabel("Residuals")
                axes[1, 1].set_ylabel("Frequency")
                axes[1, 1].set_title("Distribution of Residuals")
                axes[1, 1].spines[["top", "right"]].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig, width="content")
                plt.close(fig)

                st.info("**Model Diagnostics:** Check residual plots for patterns. Random scatter suggests good fit.")

    else:  # Multiple Linear Regression
        st.subheader("Multiple Linear Regression")
        st.markdown("Fit a linear model: **y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε**")

        y_col = st.selectbox("Dependent variable (Y)", num_cols, key="mreg_y")
        x_cols = st.multiselect("Independent variables (X)", [c for c in num_cols if c != y_col],
                                default=[c for c in num_cols if c != y_col][:min(3, len(num_cols)-1)])

        if not x_cols:
            st.info("Select at least one independent variable.")
            st.stop()

        if st.button("Fit Model"):
            data_reg = df[[y_col] + x_cols].dropna()

            if len(data_reg) < len(x_cols) + 2:
                st.error(f"Need at least {len(x_cols) + 2} observations.")
            else:
                X = data_reg[x_cols].values
                y = data_reg[y_col].values

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                # Model parameters
                beta0 = model.intercept_
                betas = model.coef_
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)

                # Adjusted R²
                n = len(data_reg)
                p = len(x_cols)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

                # Display results
                col1, col2, col3 = st.columns(3)
                col1.metric("R²", f"{r2:.4f}")
                col2.metric("Adjusted R²", f"{adj_r2:.4f}")
                col3.metric("RMSE", f"{rmse:.4f}")

                st.write(f"**Intercept (β₀):** {beta0:.4f}")
                st.write("**Coefficients:**")
                coef_df = pd.DataFrame({"Variable": x_cols, "Coefficient (β)": betas})
                st.dataframe(coef_df, width="stretch", hide_index=True)

                # Build equation string
                eq_parts = [f"{beta0:.4f}"]
                for var, coef in zip(x_cols, betas):
                    sign = "+" if coef >= 0 else "-"
                    eq_parts.append(f"{sign} {abs(coef):.4f} × {var}")
                st.write(f"**Regression Equation:** {y_col} = " + " ".join(eq_parts))

                # Residual plots
                residuals = y - y_pred

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Residuals vs Fitted
                axes[0].scatter(y_pred, residuals, alpha=0.6, color="#4c9be8", s=30)
                axes[0].axhline(0, color="red", linestyle="--", lw=2)
                axes[0].set_xlabel("Fitted values")
                axes[0].set_ylabel("Residuals")
                axes[0].set_title("Residuals vs Fitted")
                axes[0].spines[["top", "right"]].set_visible(False)

                # Q-Q plot
                (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
                axes[1].scatter(osm, osr, color="#4c9be8", s=20, alpha=0.7)
                axes[1].plot([min(osm), max(osm)],
                             [slope * min(osm) + intercept, slope * max(osm) + intercept],
                             "r--", lw=2)
                axes[1].set_xlabel("Theoretical Quantiles")
                axes[1].set_ylabel("Sample Quantiles")
                axes[1].set_title("Q-Q Plot (Residuals)")
                axes[1].spines[["top", "right"]].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig, width="content")
                plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Confidence Intervals":
    st.header("Confidence Intervals",
              help="Compute CIs for a mean, difference of means, proportion, or via "
                   "bootstrap resampling at your chosen confidence level.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found.")
        st.stop()

    ci_type = st.selectbox("Select CI Type", [
        "CI for Mean (Single Sample)",
        "CI for Difference of Means (Two Samples)",
        "CI for Proportion",
        "Bootstrap CI",
    ])

    conf_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
    alpha = 1 - conf_level

    st.markdown("---")

    if ci_type == "CI for Mean (Single Sample)":
        st.subheader("Confidence Interval for Mean")
        st.markdown(f"Construct a {conf_level*100:.0f}% CI for the population mean μ.")

        col = st.selectbox("Select column", num_cols)

        if st.button("Calculate CI"):
            data = df[col].dropna()

            if len(data) < 2:
                st.error("Need at least 2 observations.")
            else:
                mean = data.mean()
                std = data.std(ddof=1)
                n = len(data)
                se = std / np.sqrt(n)

                # t critical value
                t_crit = stats.t.ppf(1 - alpha/2, n - 1)
                margin = t_crit * se
                ci_lower = mean - margin
                ci_upper = mean + margin

                st.write(f"**Sample Mean:** {mean:.4f}")
                st.write(f"**Sample Std Dev:** {std:.4f}")
                st.write(f"**Sample Size:** {n}")
                st.write(f"**Standard Error:** {se:.4f}")
                st.write(f"**t-critical value ({conf_level*100:.0f}%):** {t_crit:.4f}")
                st.write(f"**Margin of Error:** {margin:.4f}")

                st.success(f"**{conf_level*100:.0f}% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]")
                st.info(f"We are {conf_level*100:.0f}% confident that the true population mean lies between {ci_lower:.4f} and {ci_upper:.4f}.")

    elif ci_type == "CI for Difference of Means (Two Samples)":
        st.subheader("CI for Difference of Means")
        st.markdown(f"Construct a {conf_level*100:.0f}% CI for μ₁ - μ₂.")

        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if not cat_cols:
            st.error("Need a categorical column to define groups.")
            st.stop()

        col = st.selectbox("Numeric column", num_cols)
        grp_col = st.selectbox("Grouping column", cat_cols)

        if st.button("Calculate CI"):
            groups = df[grp_col].dropna().unique()
            if len(groups) < 2:
                st.error("Need at least 2 groups.")
            else:
                g1, g2 = groups[:2]
                data1 = df[df[grp_col] == g1][col].dropna()
                data2 = df[df[grp_col] == g2][col].dropna()

                if len(data1) < 2 or len(data2) < 2:
                    st.error("Each group needs at least 2 observations.")
                else:
                    mean1, mean2 = data1.mean(), data2.mean()
                    std1, std2 = data1.std(ddof=1), data2.std(ddof=1)
                    n1, n2 = len(data1), len(data2)

                    # Welch's SE
                    se = np.sqrt(std1**2/n1 + std2**2/n2)

                    # Welch-Satterthwaite degrees of freedom
                    df_welch = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
                    t_crit = stats.t.ppf(1 - alpha/2, df_welch)

                    diff = mean1 - mean2
                    margin = t_crit * se
                    ci_lower = diff - margin
                    ci_upper = diff + margin

                    st.write(f"**Group 1 ({g1}):** Mean = {mean1:.4f}, SD = {std1:.4f}, n = {n1}")
                    st.write(f"**Group 2 ({g2}):** Mean = {mean2:.4f}, SD = {std2:.4f}, n = {n2}")
                    st.write(f"**Difference (μ₁ - μ₂):** {diff:.4f}")
                    st.write(f"**Standard Error:** {se:.4f}")
                    st.write(f"**Degrees of Freedom (Welch):** {df_welch:.2f}")

                    st.success(f"**{conf_level*100:.0f}% CI for Difference:** [{ci_lower:.4f}, {ci_upper:.4f}]")

                    if ci_lower > 0:
                        st.info(f"Since the CI is entirely above 0, {g1} has significantly higher mean than {g2}.")
                    elif ci_upper < 0:
                        st.info(f"Since the CI is entirely below 0, {g1} has significantly lower mean than {g2}.")
                    else:
                        st.info("The CI includes 0, suggesting no significant difference between groups.")

    elif ci_type == "CI for Proportion":
        st.subheader("CI for Proportion")
        st.markdown(f"Construct a {conf_level*100:.0f}% CI for population proportion p.")

        n_input = st.number_input("Sample size (n)", min_value=1, value=100, step=1)
        x_input = st.number_input("Number of successes (x)", min_value=0, max_value=n_input, value=50, step=1)

        if st.button("Calculate CI"):
            n = int(n_input)
            x = int(x_input)
            p_hat = x / n

            # Wilson score interval (more accurate than normal approximation)
            z_crit = stats.norm.ppf(1 - alpha/2)
            denominator = 1 + z_crit**2 / n
            center = (p_hat + z_crit**2 / (2*n)) / denominator
            margin = z_crit * np.sqrt(p_hat*(1-p_hat)/n + z_crit**2/(4*n**2)) / denominator

            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)

            st.write(f"**Sample Proportion (p̂):** {p_hat:.4f}")
            st.write(f"**Sample Size:** {n}")
            st.write(f"**z-critical value:** {z_crit:.4f}")

            st.success(f"**{conf_level*100:.0f}% CI for Proportion:** [{ci_lower:.4f}, {ci_upper:.4f}]")
            st.info(f"We are {conf_level*100:.0f}% confident that the true population proportion is between {ci_lower:.4f} ({ci_lower*100:.2f}%) and {ci_upper:.4f} ({ci_upper*100:.2f}%).")

    elif ci_type == "Bootstrap CI":
        st.subheader("Bootstrap Confidence Interval")
        st.markdown("Non-parametric CI using bootstrap resampling.")

        col = st.selectbox("Select column", num_cols)
        stat_func = st.selectbox("Statistic", ["Mean", "Median", "Standard Deviation"])
        n_bootstrap = st.slider("Bootstrap samples", 1000, 10000, 5000, 1000)

        if st.button("Calculate Bootstrap CI"):
            data = df[col].dropna()

            if len(data) < 10:
                st.error("Need at least 10 observations.")
            else:
                stat_map = {"Mean": np.mean, "Median": np.median, "Standard Deviation": np.std}
                stat_function = stat_map[stat_func]

                # Bootstrap
                bootstrap_stats = []
                np.random.seed(42)
                for _ in range(n_bootstrap):
                    sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_stats.append(stat_function(sample))

                bootstrap_stats = np.array(bootstrap_stats)

                # Percentile method
                ci_lower = np.percentile(bootstrap_stats, (alpha/2)*100)
                ci_upper = np.percentile(bootstrap_stats, (1-alpha/2)*100)

                observed_stat = stat_function(data)

                st.write(f"**Observed {stat_func}:** {observed_stat:.4f}")
                st.write(f"**Bootstrap Samples:** {n_bootstrap}")

                st.success(f"**{conf_level*100:.0f}% Bootstrap CI:** [{ci_lower:.4f}, {ci_upper:.4f}]")

                # Histogram of bootstrap distribution
                fig, ax = nice_fig((8, 4))
                ax.hist(bootstrap_stats, bins=50, alpha=0.7, color="#4c9be8", edgecolor="white")
                ax.axvline(observed_stat, color="red", linestyle="--", lw=2, label=f"Observed = {observed_stat:.4f}")
                ax.axvline(ci_lower, color="green", linestyle="--", lw=2, label=f"CI Lower = {ci_lower:.4f}")
                ax.axvline(ci_upper, color="green", linestyle="--", lw=2, label=f"CI Upper = {ci_upper:.4f}")
                ax.set_xlabel(f"Bootstrap {stat_func}")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Bootstrap Distribution ({n_bootstrap} samples)")
                ax.legend()
                st.pyplot(fig, width="content")
                plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# NON-PARAMETRIC TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Non-Parametric Tests":
    st.header("Non-Parametric Tests",
              help="Distribution-free alternatives: Mann-Whitney U, Wilcoxon signed-rank, "
                   "and Kruskal-Wallis tests for when normality is violated.")
    st.markdown("Distribution-free alternatives when data violates normality assumptions.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if not num_cols:
        st.error("No numeric columns found.")
        st.stop()

    test_type = st.selectbox("Select Test", [
        "Mann-Whitney U Test (2 independent samples)",
        "Wilcoxon Signed-Rank Test (paired samples)",
        "Kruskal-Wallis Test (3+ independent samples)",
    ])

    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    st.markdown("---")

    if test_type == "Mann-Whitney U Test (2 independent samples)":
        st.subheader("Mann-Whitney U Test")
        st.markdown("**Non-parametric alternative to two-sample t-test.** Tests if two independent samples come from the same distribution.")

        col = st.selectbox("Numeric column", num_cols)
        if not cat_cols:
            st.error("Need a categorical column to define groups.")
            st.stop()

        grp_col = st.selectbox("Grouping column", cat_cols)

        if st.button("Run Test"):
            groups = df[grp_col].dropna().unique()
            if len(groups) < 2:
                st.error("Need at least 2 groups.")
            else:
                g1, g2 = groups[:2]
                data1 = df[df[grp_col] == g1][col].dropna()
                data2 = df[df[grp_col] == g2][col].dropna()

                if len(data1) < 1 or len(data2) < 1:
                    st.error("Each group needs at least 1 observation.")
                else:
                    u_stat, p_val = mannwhitneyu(data1, data2, alternative="two-sided")

                    st.write(f"**Group 1 ({g1}):** Median = {data1.median():.4f}, n = {len(data1)}")
                    st.write(f"**Group 2 ({g2}):** Median = {data2.median():.4f}, n = {len(data2)}")
                    st.write(f"**U-statistic:** {u_stat:.4f}")
                    st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                    if p_val < alpha:
                        st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                                  "Significant difference between groups.")
                    else:
                        st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                               "No significant difference.")

                    st.caption("**Use when:** Data is ordinal, skewed, or has outliers (alternative to t-test).")

    elif test_type == "Wilcoxon Signed-Rank Test (paired samples)":
        st.subheader("Wilcoxon Signed-Rank Test")
        st.markdown("**Non-parametric alternative to paired t-test.** Tests if paired samples have different distributions.")

        col1 = st.selectbox("First measurement", num_cols, key="wilc1")
        col2 = st.selectbox("Second measurement", num_cols, key="wilc2")

        if st.button("Run Test"):
            valid_idx = df[[col1, col2]].dropna().index
            data1 = df.loc[valid_idx, col1]
            data2 = df.loc[valid_idx, col2]

            if len(data1) < 2:
                st.error("Need at least 2 paired observations.")
            else:
                w_stat, p_val = wilcoxon(data1, data2, alternative="two-sided")
                diff = data1 - data2

                st.write(f"**Median difference:** {diff.median():.4f}")
                st.write(f"**Number of pairs:** {len(data1)}")
                st.write(f"**W-statistic:** {w_stat:.4f}")
                st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                if p_val < alpha:
                    st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                              "Significant difference between paired measurements.")
                else:
                    st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                           "No significant difference.")

                st.caption("**Use when:** Paired data is non-normal or ordinal (alternative to paired t-test).")

    elif test_type == "Kruskal-Wallis Test (3+ independent samples)":
        st.subheader("Kruskal-Wallis Test")
        st.markdown("**Non-parametric alternative to one-way ANOVA.** Tests if 3+ independent samples come from the same distribution.")

        col = st.selectbox("Numeric column", num_cols)
        if not cat_cols:
            st.error("Need a categorical column to define groups.")
            st.stop()

        grp_col = st.selectbox("Grouping column", cat_cols)

        if st.button("Run Test"):
            groups = df[grp_col].dropna().unique()
            if len(groups) < 2:
                st.error("Need at least 2 groups.")
            else:
                group_data = [df[df[grp_col] == g][col].dropna() for g in groups]
                group_data = [g for g in group_data if len(g) > 0]

                if len(group_data) < 2:
                    st.error("Need at least 2 non-empty groups.")
                else:
                    h_stat, p_val = kruskal(*group_data)

                    st.write(f"**H-statistic:** {h_stat:.4f}")
                    st.write(f"**p-value:** {p_val:.4f} {p_stars(p_val)}")

                    # Group summaries
                    for idx, g in enumerate(groups[:len(group_data)]):
                        data_g = group_data[idx]
                        st.write(f"**{g}:** Median = {data_g.median():.4f}, n = {len(data_g)}")

                    if p_val < alpha:
                        st.success(f":material/check_circle: **Reject H₀** (p = {p_val:.4f} < α = {alpha}). " +
                                  "At least one group differs significantly.")
                    else:
                        st.info(f":material/cancel: **Fail to reject H₀** (p = {p_val:.4f} ≥ α = {alpha}). " +
                               "No significant differences among groups.")

                    st.caption("**Use when:** Data is non-normal, ordinal, or has outliers (alternative to ANOVA).")

# ══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Data Transformations":
    st.header("Data Transformations",
              help="Apply log, square-root, Box-Cox, z-score or min-max transforms and "
                   "compare distributions and normality before/after.")
    st.markdown("Transform data to meet statistical assumptions or improve model performance.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found.")
        st.stop()

    transform_type = st.selectbox("Select Transformation", [
        "Log Transformation",
        "Square Root Transformation",
        "Box-Cox Transformation",
        "Z-Score Standardization",
        "Min-Max Normalization",
    ])

    col = st.selectbox("Select column to transform", num_cols)
    data_orig = df[col].dropna()

    if len(data_orig) < 2:
        st.error("Need at least 2 observations.")
        st.stop()

    st.markdown("---")

    if transform_type == "Log Transformation":
        st.subheader("Log Transformation")
        st.markdown("**y = log(x)** — Reduces right skewness. Requires positive values.")

        if data_orig.min() <= 0:
            st.warning(f"Column contains non-positive values (min = {data_orig.min():.4f}). Adding constant to make all values positive.")
            shift = abs(data_orig.min()) + 1
            data_transformed = np.log(data_orig + shift)
            st.info(f"Applied: log(x + {shift:.2f})")
        else:
            data_transformed = np.log(data_orig)

    elif transform_type == "Square Root Transformation":
        st.subheader("Square Root Transformation")
        st.markdown("**y = √x** — Reduces right skewness (milder than log). Requires non-negative values.")

        if data_orig.min() < 0:
            st.warning(f"Column contains negative values (min = {data_orig.min():.4f}). Adding constant.")
            shift = abs(data_orig.min())
            data_transformed = np.sqrt(data_orig + shift)
            st.info(f"Applied: √(x + {shift:.2f})")
        else:
            data_transformed = np.sqrt(data_orig)

    elif transform_type == "Box-Cox Transformation":
        st.subheader("Box-Cox Transformation")
        st.markdown("**Optimal power transformation** to achieve normality. Requires positive values.")

        if data_orig.min() <= 0:
            st.error(f"Box-Cox requires positive values. Min = {data_orig.min():.4f}. Try log or sqrt first.")
            st.stop()
        else:
            data_transformed, lambda_param = stats.boxcox(data_orig)
            st.info(f"**Optimal λ = {lambda_param:.4f}**")
            st.caption(f"λ=0 → log, λ=0.5 → sqrt, λ=1 → no transform")

    elif transform_type == "Z-Score Standardization":
        st.subheader("Z-Score Standardization")
        st.markdown("**z = (x - μ) / σ** — Centers data at 0 with standard deviation 1.")

        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_orig.values.reshape(-1, 1)).flatten()

        st.write(f"**Original Mean:** {data_orig.mean():.4f}")
        st.write(f"**Original Std Dev:** {data_orig.std():.4f}")
        st.write(f"**Transformed Mean:** {data_transformed.mean():.6f}")
        st.write(f"**Transformed Std Dev:** {data_transformed.std():.6f}")

    elif transform_type == "Min-Max Normalization":
        st.subheader("Min-Max Normalization")
        st.markdown("**x' = (x - min) / (max - min)** — Scales data to [0, 1] range.")

        scaler = MinMaxScaler()
        data_transformed = scaler.fit_transform(data_orig.values.reshape(-1, 1)).flatten()

        st.write(f"**Original Range:** [{data_orig.min():.4f}, {data_orig.max():.4f}]")
        st.write(f"**Transformed Range:** [{data_transformed.min():.6f}, {data_transformed.max():.6f}]")

    # Comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original histogram
    axes[0, 0].hist(data_orig, bins=30, alpha=0.7, color="#4c9be8", edgecolor="white")
    axes[0, 0].set_title(f"Original Data — {col}")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].spines[["top", "right"]].set_visible(False)

    # Transformed histogram
    axes[0, 1].hist(data_transformed, bins=30, alpha=0.7, color="#e89b4c", edgecolor="white")
    axes[0, 1].set_title(f"Transformed Data ({transform_type})")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].spines[["top", "right"]].set_visible(False)

    # Original Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(data_orig, dist="norm")
    axes[1, 0].scatter(osm, osr, color="#4c9be8", s=20, alpha=0.7)
    axes[1, 0].plot([min(osm), max(osm)],
                    [slope * min(osm) + intercept, slope * max(osm) + intercept],
                    "r--", lw=2)
    axes[1, 0].set_title("Q-Q Plot (Original)")
    axes[1, 0].set_xlabel("Theoretical Quantiles")
    axes[1, 0].set_ylabel("Sample Quantiles")
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    # Transformed Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(data_transformed, dist="norm")
    axes[1, 1].scatter(osm, osr, color="#e89b4c", s=20, alpha=0.7)
    axes[1, 1].plot([min(osm), max(osm)],
                    [slope * min(osm) + intercept, slope * max(osm) + intercept],
                    "r--", lw=2)
    axes[1, 1].set_title("Q-Q Plot (Transformed)")
    axes[1, 1].set_xlabel("Theoretical Quantiles")
    axes[1, 1].set_ylabel("Sample Quantiles")
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)

    # Normality tests comparison
    st.subheader("Normality Tests Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Data:**")
        norm_orig = normality_table(pd.Series(data_orig))
        st.dataframe(norm_orig, width="stretch", hide_index=True)

    with col2:
        st.write("**Transformed Data:**")
        norm_trans = normality_table(pd.Series(data_transformed))
        st.dataframe(norm_trans, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING METHODS
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Sampling Methods":
    st.header("Sampling Methods",
              help="Demonstrate simple random, stratified and systematic sampling, plus a "
                   "sample-size calculator.")
    st.markdown("Demonstrate different sampling techniques and sample size calculations.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first.")
        st.stop()

    sampling_type = st.selectbox("Select Sampling Method", [
        "Simple Random Sampling",
        "Stratified Sampling",
        "Systematic Sampling",
        "Sample Size Calculator",
    ])

    st.markdown("---")

    if sampling_type == "Simple Random Sampling":
        st.subheader("Simple Random Sampling (SRS)")
        st.markdown("Each observation has equal probability of selection.")

        sample_size = st.slider("Sample size", 1, min(len(df), 1000), min(100, len(df)//2))

        if st.button("Draw Sample"):
            sample = df.sample(n=sample_size, random_state=42)

            st.success(f"Drew {sample_size} observations from population of {len(df)}.")
            st.write("**Sample Preview:**")
            st.dataframe(sample.head(10), width="stretch")

            # Compare sample vs population for numeric columns
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                comp_col = st.selectbox("Compare distribution", num_cols)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                axes[0].hist(df[comp_col].dropna(), bins=30, alpha=0.7, color="#4c9be8", edgecolor="white")
                axes[0].set_title(f"Population — {comp_col}")
                axes[0].set_xlabel("Value")
                axes[0].set_ylabel("Frequency")
                axes[0].axvline(df[comp_col].mean(), color="red", linestyle="--", lw=2, label=f"Mean = {df[comp_col].mean():.2f}")
                axes[0].legend()
                axes[0].spines[["top", "right"]].set_visible(False)

                axes[1].hist(sample[comp_col].dropna(), bins=30, alpha=0.7, color="#e89b4c", edgecolor="white")
                axes[1].set_title(f"Sample (n={sample_size}) — {comp_col}")
                axes[1].set_xlabel("Value")
                axes[1].set_ylabel("Frequency")
                axes[1].axvline(sample[comp_col].mean(), color="red", linestyle="--", lw=2, label=f"Mean = {sample[comp_col].mean():.2f}")
                axes[1].legend()
                axes[1].spines[["top", "right"]].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig, width="content")
                plt.close(fig)

    elif sampling_type == "Stratified Sampling":
        st.subheader("Stratified Sampling")
        st.markdown("Sample proportionally from each stratum (group).")

        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if not cat_cols:
            st.error("Need a categorical column to define strata.")
            st.stop()

        strata_col = st.selectbox("Stratification column", cat_cols)
        sample_frac = st.slider("Sample fraction", 0.05, 0.50, 0.20, 0.05)

        if st.button("Draw Stratified Sample"):
            sample = df.groupby(strata_col, group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=42))

            st.success(f"Drew stratified sample: {len(sample)} observations ({sample_frac*100:.0f}% from each stratum).")

            # Stratum distribution comparison
            pop_counts = df[strata_col].value_counts()
            sample_counts = sample[strata_col].value_counts()

            comparison = pd.DataFrame({
                "Stratum": pop_counts.index,
                "Population Count": pop_counts.values,
                "Sample Count": sample_counts.values,
                "Population %": (pop_counts.values / len(df) * 100).round(2),
                "Sample %": (sample_counts.values / len(sample) * 100).round(2),
            })

            st.write("**Stratification Comparison:**")
            st.dataframe(comparison, width="stretch", hide_index=True)

    elif sampling_type == "Systematic Sampling":
        st.subheader("Systematic Sampling")
        st.markdown("Select every kth observation from ordered list.")

        sample_size = st.slider("Desired sample size", 1, min(len(df), 1000), min(100, len(df)//2))
        k = len(df) // sample_size

        st.write(f"**Sampling interval (k):** {k}")
        st.write(f"**Starting point:** Random between 1 and {k}")

        if st.button("Draw Systematic Sample"):
            start = np.random.randint(0, k)
            indices = list(range(start, len(df), k))
            sample = df.iloc[indices]

            st.success(f"Drew {len(sample)} observations using systematic sampling (every {k}th observation).")
            st.write("**Sample Preview:**")
            st.dataframe(sample.head(10), width="stretch")

    elif sampling_type == "Sample Size Calculator":
        st.subheader("Sample Size Calculator")
        st.markdown("Calculate required sample size for estimating a population mean or proportion.")

        calc_type = st.radio("Calculate for:", ["Mean", "Proportion"])

        if calc_type == "Mean":
            st.write("**Formula:** n = (z² × σ²) / E²")

            conf_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            sigma = st.number_input("Population Std Dev (σ)", min_value=0.1, value=10.0, step=0.1)
            margin = st.number_input("Margin of Error (E)", min_value=0.1, value=2.0, step=0.1)

            alpha = 1 - conf_level
            z_crit = stats.norm.ppf(1 - alpha/2)

            n_required = np.ceil((z_crit**2 * sigma**2) / margin**2)

            st.write(f"**z-critical value ({conf_level*100:.0f}%):** {z_crit:.4f}")
            st.success(f"**Required Sample Size:** {int(n_required)}")

            st.info(f"To estimate the population mean with {conf_level*100:.0f}% confidence " +
                   f"and margin of error ±{margin}, you need **{int(n_required)} observations**.")

        else:  # Proportion
            st.write("**Formula:** n = (z² × p × (1-p)) / E²")

            conf_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            p_estimate = st.slider("Estimated proportion (p)", 0.01, 0.99, 0.50, 0.01)
            margin = st.slider("Margin of Error (E)", 0.01, 0.20, 0.05, 0.01)

            alpha = 1 - conf_level
            z_crit = stats.norm.ppf(1 - alpha/2)

            n_required = np.ceil((z_crit**2 * p_estimate * (1 - p_estimate)) / margin**2)

            st.write(f"**z-critical value ({conf_level*100:.0f}%):** {z_crit:.4f}")
            st.success(f"**Required Sample Size:** {int(n_required)}")

            st.info(f"To estimate the population proportion with {conf_level*100:.0f}% confidence " +
                   f"and margin of error ±{margin*100:.1f}%, you need **{int(n_required)} observations**.")
            st.caption("Using p=0.5 (maximum variance) gives most conservative estimate.")

# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "Generate PDF Report":
    st.header(":material/description: Generate PDF Report",
              help="Preview and edit collected report items, configure title/author, "
                   "auto-include sections, then export a polished PDF.")
    st.markdown("Build and preview your custom report before exporting to PDF.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to **Data Explorer** first to load data.")
        st.stop()

    # Show report builder status
    n_items = len(st.session_state.report_items)
    col1, col2, col3 = st.columns(3)
    col1.metric("Report Items", n_items)
    col2.metric("Dataset Rows", f"{len(df):,}")
    col3.metric("Dataset Columns", len(df.columns))

    st.markdown("---")

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs([":material/edit_note: Report Preview & Editor", ":material/settings: Report Configuration", ":material/bar_chart: Auto-Include Sections"])
    
    # TAB 1: Report Preview & Editor
    with tab1:
        st.subheader("Report Items Preview")
        
        if n_items == 0:
            st.info(":material/search: No items added to report yet. Use 'Add to Report' buttons in other tools to build your report.")
            st.markdown("**Available tools with 'Add to Report' functionality:**")
            st.markdown("""
            - Descriptive Statistics → Summary tables
            - Descriptive Statistics → Correlation matrices
            - Data Visualization → Charts and plots
            - Statistical Inference → Test results
            - Regression Analysis → Model results
            - And more...  
            """)
        else:
            st.success(f":material/check_circle: {n_items} item(s) ready to include in your report")
            
            # Display each report item with options to remove
            for idx, item in enumerate(st.session_state.report_items):
                with st.expander(f":material/description: {idx+1}. {item['title']} (Added: {item['timestamp']})", expanded=True):
                    col_preview, col_actions = st.columns([4, 1])
                    
                    with col_preview:
                        if item['type'] == 'descriptive_stats':
                            st.markdown("**Type:** Descriptive Statistics Table")
                            st.dataframe(pd.DataFrame(item['data']), width=600)
                        
                        elif item['type'] == 'correlation_heatmap':
                            st.markdown("**Type:** Correlation Heatmap")
                            st.image(item['image'], width=500)
                        
                        elif item['type'] == 'plot':
                            st.markdown(f"**Type:** {item.get('plot_type', 'Plot')}")
                            st.image(item['image'], width=500)
                        
                        elif item['type'] == 'test_result':
                            st.markdown(f"**Type:** Statistical Test - {item.get('test_name', 'Test')}")
                            st.json(item.get('results', {}))
                        
                        else:
                            st.markdown(f"**Type:** {item['type']}")
                            st.json(item)
                    
                    with col_actions:
                        if st.button(":material/delete: Remove", key=f"remove_{idx}"):
                            st.session_state.report_items.pop(idx)
                            st.rerun()
                        
                        # Move up/down buttons
                        if idx > 0:
                            if st.button(":material/arrow_upward: Move Up", key=f"up_{idx}"):
                                st.session_state.report_items[idx], st.session_state.report_items[idx-1] = \
                                    st.session_state.report_items[idx-1], st.session_state.report_items[idx]
                                st.rerun()
                        
                        if idx < n_items - 1:
                            if st.button(":material/arrow_downward: Move Down", key=f"down_{idx}"):
                                st.session_state.report_items[idx], st.session_state.report_items[idx+1] = \
                                    st.session_state.report_items[idx+1], st.session_state.report_items[idx]
                                st.rerun()
            
            st.markdown("---")
            
            # Custom text/notes section
            st.subheader(":material/add: Add Custom Notes")
            custom_note = st.text_area("Add custom notes or commentary to your report")
            if st.button("Add Note to Report"):
                if custom_note.strip():
                    st.session_state.report_items.append({
                        "type": "custom_note",
                        "title": "Custom Note",
                        "content": custom_note,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    st.success(":material/check_circle: Note added!")
                    st.rerun()
    
    # TAB 2: Report Configuration
    with tab2:
        st.subheader("Report Configuration")
        
        st.session_state.report_title = st.text_input(
            "Report Title", 
            value=st.session_state.report_title
        )
        st.session_state.report_author = st.text_input(
            "Author Name", 
            value=st.session_state.report_author
        )
        
        st.markdown("**PDF Layout Options:**")
        page_size = st.selectbox("Page Size", ["Letter (8.5×11)", "A4"])
        include_page_numbers = st.checkbox("Include page numbers", value=True)
    
    # TAB 3: Auto-Include Sections
    with tab3:
        st.subheader("Automatic Dataset Analysis Sections")
        st.markdown("These sections will be auto-generated from your current dataset:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_summary = st.checkbox("Dataset Summary", value=True)
            include_descriptive = st.checkbox("Descriptive Statistics", value=False)
            include_correlation = st.checkbox("Correlation Matrix", value=False)
            include_histograms = st.checkbox("Histograms (all numeric columns)", value=False)
        
        with col2:
            include_missing = st.checkbox("Missing Values Analysis", value=True)
            include_boxplots = st.checkbox("Box Plots (all numeric columns)", value=False)
            include_distributions = st.checkbox("Distribution Plots", value=False)
        
        st.info(":material/lightbulb: **Tip:** These auto-generated sections will be added AFTER your custom report items from Tab 1.")

    st.markdown("---")

    # Generate PDF button
    st.subheader(":material/palette: Generate Final PDF")
    col_gen1, col_gen2 = st.columns([3, 1])
    with col_gen1:
        st.markdown(f"**Report will include:** {n_items} custom item(s) + auto-generated sections")
    with col_gen2:
        generate_button = st.button(":material/download: Generate & Download PDF", type="primary", use_container_width=True)
    
    if generate_button:
        with st.spinner("Generating PDF report... This may take a moment."):
            try:
                # Create PDF in memory
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter,
                                      topMargin=0.75*inch, bottomMargin=0.75*inch,
                                      leftMargin=0.75*inch, rightMargin=0.75*inch)
                
                # Container for PDF elements
                story = []
                styles = getSampleStyleSheet()
                
                # Custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#1f77b4'),
                    spaceAfter=30,
                    alignment=TA_CENTER
                )
                
                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=16,
                    textColor=colors.HexColor('#2c3e50'),
                    spaceAfter=12,
                    spaceBefore=12
                )
                
                # Title Page
                story.append(Spacer(1, 1.5*inch))
                story.append(Paragraph(st.session_state.report_title, title_style))
                story.append(Spacer(1, 0.3*inch))
                story.append(Paragraph(f"<b>Author:</b> {st.session_state.report_author}", styles['Normal']))
                story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                story.append(Paragraph(f"<b>Tool:</b> Data Analysis Toolkit", styles['Normal']))
                story.append(Paragraph(f"<b>Report Items:</b> {n_items} custom + auto-generated sections", styles['Normal']))
                story.append(PageBreak())
                
                # Add custom report items first
                if n_items > 0:
                    story.append(Paragraph("Custom Report Items", heading_style))
                    story.append(Spacer(1, 0.2*inch))
                    
                    for idx, item in enumerate(st.session_state.report_items):
                        story.append(Paragraph(f"{idx+1}. {item['title']}", styles['Heading3']))
                        story.append(Spacer(1, 0.1*inch))
                        
                        if item['type'] == 'descriptive_stats':
                            # Add descriptive stats table
                            df_item = pd.DataFrame(item['data'])
                            data_table = [[str(col) for col in df_item.columns]]
                            for _, row in df_item.iterrows():
                                data_table.append([str(val) for val in row.values])
                            
                            table = Table(data_table)
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c9be8')),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 9),
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ]))
                            story.append(table)
                        
                        elif item['type'] in ['correlation_heatmap', 'plot']:
                            # Add image
                            img_data = BytesIO(item['image'])
                            img = Image(img_data, width=5.5*inch, height=4*inch)
                            story.append(img)
                        
                        elif item['type'] == 'custom_note':
                            # Add custom note
                            story.append(Paragraph(item['content'], styles['Normal']))
                        
                        elif item['type'] == 'test_result':
                            # Add test results
                            results = item.get('results', {})
                            for key, value in results.items():
                                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                        
                        story.append(Spacer(1, 0.3*inch))
                        
                        # Page break after every 2 items
                        if (idx + 1) % 2 == 0 and idx < n_items - 1:
                            story.append(PageBreak())
                    
                    story.append(PageBreak())
                
                # Auto-generated sections
                story.append(Paragraph("Auto-Generated Dataset Analysis", heading_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Dataset Summary
                if include_summary:
                    story.append(Paragraph("Dataset Overview", heading_style))
                    story.append(Spacer(1, 0.2*inch))
                    
                    summary_data = [
                        ['Metric', 'Value'],
                        ['Number of Rows', f'{len(df):,}'],
                        ['Number of Columns', str(len(df.columns))],
                        ['Numeric Columns', str(len(df.select_dtypes(include=np.number).columns))],
                        ['Categorical Columns', str(len(df.select_dtypes(exclude=np.number).columns))],
                        ['Memory Usage', f'{df.memory_usage(deep=True).sum() / 1024:.2f} KB'],
                    ]
                    
                    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
                    summary_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c9be8')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]))
                    story.append(summary_table)
                    story.append(Spacer(1, 0.3*inch))
                
                # Missing Values
                if include_missing:
                    story.append(Paragraph("Missing Values Analysis", heading_style))
                    story.append(Spacer(1, 0.2*inch))
                    
                    missing_df = df.isnull().sum()
                    missing_df = missing_df[missing_df > 0]
                    
                    if len(missing_df) > 0:
                        missing_data = [['Column', 'Missing Count', 'Percentage']]
                        for col, count in missing_df.items():
                            pct = (count / len(df) * 100)
                            missing_data.append([col, str(count), f'{pct:.2f}%'])
                        
                        missing_table = Table(missing_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
                        missing_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 11),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                        ]))
                        story.append(missing_table)
                    else:
                        story.append(Paragraph(":material/check_circle: No missing values found in the dataset.", styles['Normal']))
                    
                    story.append(Spacer(1, 0.3*inch))
                
                # Descriptive Statistics
                if include_descriptive:
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        story.append(PageBreak())
                        story.append(Paragraph("Descriptive Statistics", heading_style))
                        story.append(Spacer(1, 0.2*inch))
                        
                        desc = df[num_cols].describe().round(3).T
                        
                        # Limit to first 10 columns if too many
                        if len(desc) > 10:
                            desc = desc.head(10)
                            story.append(Paragraph("<i>Showing first 10 numeric columns</i>", styles['Italic']))
                            story.append(Spacer(1, 0.1*inch))
                        
                        desc_data = [['Column', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']]
                        for col in desc.index:
                            row = [col] + [f"{desc.loc[col, stat]:.2f}" for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                            desc_data.append(row)
                        
                        desc_table = Table(desc_data, colWidths=[1.2*inch] + [0.7*inch]*7)
                        desc_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                        ]))
                        story.append(desc_table)
                        story.append(Spacer(1, 0.3*inch))
                
                # Correlation Matrix
                if include_correlation:
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if len(num_cols) >= 2:
                        story.append(PageBreak())
                        story.append(Paragraph("Correlation Analysis", heading_style))
                        story.append(Spacer(1, 0.2*inch))
                        
                        # Generate correlation heatmap
                        corr = df[num_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                        
                        ax.set_xticks(np.arange(len(corr.columns)))
                        ax.set_yticks(np.arange(len(corr.index)))
                        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(corr.index, fontsize=8)
                        
                        # Add correlation values
                        for i in range(len(corr.columns)):
                            for j in range(len(corr.columns)):
                                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                                             ha="center", va="center", color="black", fontsize=7)
                        
                        plt.colorbar(im, ax=ax)
                        ax.set_title("Correlation Heatmap")
                        plt.tight_layout()
                        
                        # Save to buffer
                        img_buffer = BytesIO()
                        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        img_buffer.seek(0)
                        
                        # Add to PDF
                        img = Image(img_buffer, width=5.5*inch, height=4.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.3*inch))
                
                # Distribution Plots
                if include_histograms:
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        story.append(PageBreak())
                        story.append(Paragraph("Distribution Histograms", heading_style))
                        story.append(Spacer(1, 0.2*inch))
                        
                        # Create histograms (max 6 per page)
                        for i in range(0, min(len(num_cols), 6), 2):
                            cols_to_plot = num_cols[i:i+2]
                            n_plots = len(cols_to_plot)
                            
                            fig, axes = plt.subplots(1, n_plots, figsize=(6.5, 3))
                            if n_plots == 1:
                                axes = [axes]
                            
                            for ax, col in zip(axes, cols_to_plot):
                                data_col = df[col].dropna()
                                ax.hist(data_col, bins=30, color='#4c9be8', edgecolor='white', alpha=0.7)
                                ax.set_xlabel(col, fontsize=9)
                                ax.set_ylabel('Frequency', fontsize=9)
                                ax.set_title(f'{col}\n(μ={data_col.mean():.2f}, σ={data_col.std():.2f})', fontsize=9)
                                ax.spines[['top', 'right']].set_visible(False)
                                ax.tick_params(labelsize=8)
                            
                            plt.tight_layout()
                            
                            img_buffer = BytesIO()
                            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            img_buffer.seek(0)
                            
                            img = Image(img_buffer, width=6.5*inch, height=3*inch)
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                            
                            if i + 2 < min(len(num_cols), 6):
                                story.append(Spacer(1, 0.1*inch))
                
                # Box Plots
                if include_boxplots:
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        story.append(PageBreak())
                        story.append(Paragraph("Box Plots (Outlier Detection)", heading_style))
                        story.append(Spacer(1, 0.2*inch))
                        
                        # Create box plots (max 6)
                        for i in range(0, min(len(num_cols), 6), 2):
                            cols_to_plot = num_cols[i:i+2]
                            n_plots = len(cols_to_plot)
                            
                            fig, axes = plt.subplots(1, n_plots, figsize=(6.5, 3))
                            if n_plots == 1:
                                axes = [axes]
                            
                            for ax, col in zip(axes, cols_to_plot):
                                data_col = df[col].dropna()
                                bp = ax.boxplot([data_col], tick_labels=[col], patch_artist=True)
                                bp['boxes'][0].set_facecolor('#4c9be8')
                                bp['boxes'][0].set_alpha(0.7)
                                ax.set_ylabel('Value', fontsize=9)
                                ax.set_title(col, fontsize=9)
                                ax.spines[['top', 'right']].set_visible(False)
                                ax.tick_params(labelsize=8)
                            
                            plt.tight_layout()
                            
                            img_buffer = BytesIO()
                            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            img_buffer.seek(0)
                            
                            img = Image(img_buffer, width=6.5*inch, height=3*inch)
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                
                # Footer
                story.append(PageBreak())
                story.append(Paragraph("Report Summary", heading_style))
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph(
                    f"This report was automatically generated by the Data Analysis Toolkit. "
                    f"The dataset contains {len(df):,} observations across {len(df.columns)} variables. "
                    f"All statistical analyses were performed using Python with pandas, numpy, scipy, and matplotlib.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph(
                    f"<b>Report Generation Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
                    styles['Normal']
                ))
                
                # Build PDF
                doc.build(story)
                buffer.seek(0)
                
                # Offer download
                st.success(":material/check_circle: PDF report generated successfully!")
                st.download_button(
                    label=":material/download: Download PDF Report",
                    data=buffer,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                
                st.info(":material/lightbulb: **Tip:** The PDF includes all selected sections with visualizations and statistical summaries.")
                
            except Exception as e:
                st.error(f":material/cancel: Error generating PDF: {str(e)}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif tool_key == "AI Assistant":
    st.header(":material/smart_toy: AI Data Analysis Assistant",
              help="Chat with a local Ollama model that can run analysis tools on your "
                   "dataset, remember facts, and add results to your PDF report.")

    # ---- Check Ollama availability ----
    if not _HAS_OLLAMA:
        st.error(":material/cancel: The `ollama` Python package is not installed. Run `pip install ollama`.")
        st.stop()

    available_models = get_ollama_models()
    if not available_models:
        st.error(":material/cancel: No Ollama models found or Ollama server is not running. "
                 "Start Ollama and pull a model (e.g. `ollama pull llama3.2`).")
        st.stop()

    df = st.session_state.get("df")

    # ═══ Two-column layout: chat (center) · control panel (right) ═══
    chat_col, panel_col = st.columns([2, 1], gap="large")

    # ───────────────────────── RIGHT CONTROL PANEL ─────────────────────────
    with panel_col:
        st.markdown("### :material/tune: Control Panel")
        p_settings, p_system, p_memory, p_tools, p_help = st.tabs(
            [":material/settings: Settings", ":material/computer: System", ":material/psychology: Memory", ":material/build: Tools", "ℹ Help"]
        )

        # ---- Settings tab ----
        with p_settings:
            default_idx = 0
            if st.session_state.ai_model in available_models:
                default_idx = available_models.index(st.session_state.ai_model)
            selected_model = st.selectbox(":material/extension: Ollama Model", available_models, index=default_idx)
            st.session_state.ai_model = selected_model

            temperature = st.slider(":material/thermostat: Temperature", 0.0, 2.0, 0.7, 0.1,
                                    help="Higher = more creative, lower = more deterministic")
            top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
            max_tokens = st.number_input("Max tokens (num_predict)", 64, 8192, 1024, 64)
            context_window = st.number_input(
                "Context window limit (num_ctx)", 512, 32768,
                int(st.session_state.ai_context_limit), 512,
                help="Maximum context size in tokens. The usage bar tracks against this limit."
            )
            st.session_state.ai_context_limit = context_window

            st.markdown("**Behavior**")
            enable_tools = st.checkbox(":material/build: Enable analysis tools", value=True,
                                       help="Let the AI call functions to analyze your data")
            include_data_context = st.checkbox(":material/attach_file: Include dataset summary", value=True)
            include_memory = st.checkbox(":material/psychology: Include memory in context", value=True,
                                         help="Inject stored memory notes into the system prompt")
            enable_thinking = st.checkbox(":material/psychology_alt: Show live thinking", value=True,
                                          help="Stream the model's reasoning live (reasoning models only)")
            auto_summarize = st.checkbox(":material/edit_note: Auto-summarize when full", value=True,
                                         help="When usage exceeds 80% of the limit, summarize old messages via AI")
            st.caption("Settings apply to the next message.")

        # ---- System tab ----
        with p_system:
            sys_stats = get_system_stats()
            if sys_stats["cpu_percent"] is not None:
                st.metric(":material/computer: CPU Usage", f"{sys_stats['cpu_percent']:.0f}%")
            else:
                st.metric(":material/computer: CPU Usage", "N/A")
            if sys_stats["ram_percent"] is not None:
                st.metric(":material/psychology: RAM Usage", f"{sys_stats['ram_percent']:.0f}%",
                          f"{sys_stats['ram_used_gb']:.1f}/{sys_stats['ram_total_gb']:.1f} GB")
            else:
                st.metric(":material/psychology: RAM Usage", "N/A")
            st.metric(":material/memory: GPU", "Apple Silicon" if platform.system() == "Darwin" else "GPU")
            st.caption(sys_stats["gpu_info"])
            st.metric(":material/chat: Messages", len(st.session_state.ai_messages))
            if st.button(":material/refresh: Refresh Stats", width="stretch"):
                st.rerun()

        # ---- Memory tab ----
        with p_memory:
            st.caption(f"{len(st.session_state.ai_memory)} note(s) stored. "
                       "The AI remembers these across chats.")
            new_mem = st.text_input("Add a memory note", key="new_memory_input",
                                    placeholder="e.g. User prefers non-parametric tests")
            new_mem_cat = st.text_input("Category", key="new_memory_cat", value="note")
            if st.button(":material/add: Add Memory", width="stretch"):
                if new_mem.strip():
                    st.session_state.ai_memory.append({
                        "content": new_mem.strip(),
                        "category": new_mem_cat.strip() or "note",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    save_ai_memory(st.session_state.ai_memory)
                    st.rerun()

            if st.session_state.ai_memory:
                st.markdown("**Stored memories:**")
                for midx, mem in enumerate(st.session_state.ai_memory):
                    rc1, rc2 = st.columns([5, 1])
                    with rc1:
                        st.markdown(f"`[{mem['category']}]` {mem['content']}  \n"
                                    f"<span style='color:gray;font-size:0.8em'>saved {mem['timestamp']}</span>",
                                    unsafe_allow_html=True)
                    with rc2:
                        if st.button(":material/delete:", key=f"del_mem_{midx}"):
                            st.session_state.ai_memory.pop(midx)
                            save_ai_memory(st.session_state.ai_memory)
                            st.rerun()
                if st.button(":material/delete: Clear All", width="stretch"):
                    st.session_state.ai_memory = []
                    save_ai_memory([])
                    st.rerun()
            else:
                st.info("No memories stored yet.")

        # ---- Tools tab ----
        with p_tools:
            st.markdown("""
The AI can call these tools on your dataset:
- **describe** — descriptive statistics
- **correlation** — Pearson correlation / matrix
- **ttest** — two-sample t-test by group
- **value_counts** — value frequency counts
- **missing_values** — missing-value report
- **plot** — draw a chart (histogram, box, scatter, bar, line, correlation) in the chat
- **remember** — save a fact to memory
- **recall** — retrieve memory notes
- **add_to_report** — add result to PDF report
            """)

        # ---- Help tab ----
        with p_help:
            st.markdown("""
**How to use this assistant**

1. Pick a model and tune settings in **:material/settings: Settings**.
2. Load a CSV in **Data Explorer** so the AI can analyze it.
3. Type a question in the chat box (center).
4. Ask it to *"remember"* preferences or *"add this to my report"*.
5. Watch the **:material/bar_chart: Context** bar — use **Summarize Now** when it fills up.

The model runs **locally** via Ollama; no data leaves your machine.
            """)

        # ---- Context usage bar (always visible) ----
        st.markdown("---")
        ctx_limit = int(st.session_state.ai_context_limit)
        used_tokens = estimate_messages_tokens(st.session_state.ai_messages)
        if include_data_context and df is not None:
            used_tokens += estimate_tokens(get_dataframe_context(df))
        if include_memory and st.session_state.ai_memory:
            used_tokens += estimate_tokens(
                "\n".join(m["content"] for m in st.session_state.ai_memory)
            )
        usage_frac = min(1.0, used_tokens / ctx_limit) if ctx_limit else 0.0
        st.markdown(f"**:material/bar_chart: Context:** ~{used_tokens:,} / {ctx_limit:,} ({usage_frac*100:.0f}%)")
        st.progress(usage_frac)
        if usage_frac >= 0.8:
            st.warning(":material/warning: Context nearly full.")
        if st.button(":material/edit_note: Summarize Now", width="stretch",
                     help="Compress older messages into a summary via AI"):
            if len(st.session_state.ai_messages) > 2 and st.session_state.ai_model:
                try:
                    with st.spinner("Summarizing conversation..."):
                        opts = {"temperature": 0.3, "num_ctx": ctx_limit}
                        summary = summarize_conversation(
                            st.session_state.ai_model,
                            st.session_state.ai_messages,
                            opts,
                        )
                        st.session_state.ai_messages = [
                            {"role": "assistant",
                             "content": ":material/content_paste: **Conversation summary (older messages compressed):**\n\n" + summary}
                        ]
                    st.success("Conversation summarized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
            else:
                st.info("Not enough conversation to summarize.")

    # ───────────────────────── CENTER CHAT AREA ─────────────────────────
    with chat_col:
        ds1, ds2 = st.columns([3, 1])
        with ds1:
            if df is None:
                st.warning(":material/warning: No dataset loaded — load one in **Data Explorer** for analysis.")
            else:
                st.success(f":material/check_circle: Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        with ds2:
            if st.button(":material/delete: Clear Chat", width="stretch"):
                st.session_state.ai_messages = []
                st.rerun()

        chat_container = st.container(height=480)

    # ---- Display chat history (inside the scrollable chat container) ----
    with chat_container:
        if not st.session_state.ai_messages:
            st.caption(":material/lightbulb: Start the conversation below. Try: *\"Describe my dataset\"* or "
                       "*\"Run a t-test on column X grouped by Y\"*.")
        for msg in st.session_state.ai_messages:
            if msg["role"] in ("user", "assistant"):
                with st.chat_message(msg["role"]):
                    if msg.get("thinking"):
                        with st.expander(":material/psychology: Reasoning", expanded=False):
                            st.markdown(msg["thinking"])
                    st.markdown(msg["content"])
                    for img in msg.get("images", []) or []:
                        st.image(img["data"], caption=img.get("caption"), width="stretch")
            elif msg["role"] == "tool":
                with st.chat_message("assistant"):
                    st.caption(f":material/build: Tool result ({msg.get('name', 'tool')}):")
                    st.code(msg["content"], language="text")

    # ---- Chat input ----
    user_input = chat_col.chat_input("Ask about your data...")

    if user_input:
        # Append user message
        st.session_state.ai_messages.append({"role": "user", "content": user_input})
        with chat_container, st.chat_message("user"):
            st.markdown(user_input)

        # Build message list for the model
        system_prompt = (
            "You are a helpful data analysis assistant integrated into a statistics toolkit. "
            "You help users understand their dataset and run statistical analyses. "
            "When the user asks about their data, use the available tools to get real results "
            "rather than guessing. Be concise and explain results in plain language. "
            "When the user asks to see, show, visualize or plot data, call the 'plot' tool so "
            "the chart is displayed in the chat. "
            "Use the 'remember' tool to save important user preferences or findings, and "
            "'add_to_report' to add results to the user's report when they ask for a report."
        )
        if include_data_context and df is not None:
            system_prompt += "\n\nCurrent dataset context:\n" + get_dataframe_context(df)
        if include_memory and st.session_state.ai_memory:
            mem_text = "\n".join(
                f"- [{m['category']}] {m['content']}" for m in st.session_state.ai_memory
            )
            system_prompt += "\n\nStored memory notes:\n" + mem_text

        # Auto-summarize if context is nearly full
        if auto_summarize:
            projected = estimate_messages_tokens(st.session_state.ai_messages) + \
                estimate_tokens(system_prompt)
            if projected > 0.8 * int(context_window) and len(st.session_state.ai_messages) > 4:
                try:
                    opts_sum = {"temperature": 0.3, "num_ctx": int(context_window)}
                    older = st.session_state.ai_messages[:-2]
                    recent = st.session_state.ai_messages[-2:]
                    summary = summarize_conversation(selected_model, older, opts_sum)
                    st.session_state.ai_messages = [
                        {"role": "assistant",
                         "content": ":material/content_paste: **Earlier conversation summary:**\n\n" + summary}
                    ] + recent
                except Exception:
                    pass

        api_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.ai_messages:
            if m["role"] in ("user", "assistant"):
                api_messages.append({"role": m["role"], "content": m["content"]})
            elif m["role"] == "tool":
                api_messages.append({"role": "tool", "content": m["content"]})

        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": int(max_tokens),
            "num_ctx": int(context_window),
        }

        with chat_container, st.chat_message("assistant"):
            log_box = st.empty()       # live logs (removed when finished)
            thinking_box = st.empty()  # streaming reasoning
            answer_box = st.empty()    # streaming answer
            image_box = st.empty()     # rendered plots

            logs = []
            st.session_state["_ai_plot_images"] = []  # reset queued plots

            def push_log(line):
                logs.append(f"• {line}")
                log_box.code("\n".join(logs), language="text")

            def stream_chat(messages, use_tools):
                """Stream an Ollama chat call; update thinking/answer boxes live.
                Returns (content, thinking, tool_calls)."""
                kwargs = {
                    "model": selected_model,
                    "messages": messages,
                    "options": options,
                    "stream": True,
                }
                if enable_thinking:
                    kwargs["think"] = True
                if use_tools:
                    kwargs["tools"] = AI_TOOL_SCHEMA

                content_acc = ""
                think_acc = ""
                collected_calls = []

                def _consume(stream_iter):
                    nonlocal content_acc, think_acc
                    for chunk in stream_iter:
                        cmsg = chunk.message
                        think_piece = getattr(cmsg, "thinking", None)
                        if think_piece:
                            think_acc += think_piece
                            thinking_box.markdown(":material/psychology: **Thinking…**\n\n" + think_acc)
                        piece = getattr(cmsg, "content", None)
                        if piece:
                            content_acc += piece
                            answer_box.markdown(content_acc + " ▌")
                        tcs = getattr(cmsg, "tool_calls", None)
                        if tcs:
                            collected_calls.extend(tcs)

                try:
                    _consume(ollama.chat(**kwargs))
                except Exception as stream_err:
                    # Retry without thinking if the model doesn't support it
                    if enable_thinking and "think" in str(stream_err).lower():
                        push_log("Model does not support thinking; retrying without it…")
                        kwargs.pop("think", None)
                        _consume(ollama.chat(**kwargs))
                    else:
                        raise

                return content_acc, think_acc, collected_calls

            try:
                push_log(f"Connecting to model: {selected_model}")
                push_log(f"Sending {len(api_messages)} message(s) · tools={'on' if enable_tools else 'off'}")
                push_log("Streaming response…")

                answer, thinking, tool_calls = stream_chat(api_messages, enable_tools)

                # Handle tool calls
                if tool_calls:
                    push_log(f"Model requested {len(tool_calls)} tool call(s)")
                    # Record the assistant's tool-call turn
                    api_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tool_calls": [
                            {"function": {"name": tc.function.name,
                                          "arguments": tc.function.arguments}}
                            for tc in tool_calls
                        ],
                    })
                    for tc in tool_calls:
                        fname = tc.function.name
                        fargs = tc.function.arguments
                        if isinstance(fargs, str):
                            try:
                                fargs = json.loads(fargs)
                            except Exception:
                                fargs = {}
                        push_log(f"Running tool: {fname}({fargs})")
                        result = run_ai_tool(fname, fargs, df)
                        push_log(f"Tool {fname} → {len(str(result))} chars returned")
                        st.session_state.ai_messages.append(
                            {"role": "tool", "name": fname, "content": str(result)}
                        )
                        api_messages.append({"role": "tool", "content": str(result)})

                    push_log("Generating final answer with tool results…")
                    answer, thinking2, _ = stream_chat(api_messages, False)
                    if thinking2:
                        thinking = (thinking + "\n\n" + thinking2) if thinking else thinking2

                # Finalize: render clean answer, collapse thinking, remove logs
                answer_box.markdown(answer)

                # Render any plots the AI generated this turn
                plot_imgs = st.session_state.get("_ai_plot_images", [])
                if plot_imgs:
                    with image_box.container():
                        for img in plot_imgs:
                            st.image(img["data"], caption=img.get("caption"),
                                     width="stretch")

                if thinking:
                    thinking_box.empty()
                    with thinking_box.container():
                        with st.expander(":material/psychology: Reasoning", expanded=False):
                            st.markdown(thinking)
                else:
                    thinking_box.empty()
                log_box.empty()  # remove logs when finished

                st.session_state.ai_messages.append(
                    {"role": "assistant", "content": answer, "thinking": thinking,
                     "images": plot_imgs}
                )
                st.session_state["_ai_plot_images"] = []

            except Exception as e:
                log_box.empty()
                thinking_box.empty()
                answer_box.error(f":material/cancel: Error communicating with Ollama: {e}")

        st.rerun()


