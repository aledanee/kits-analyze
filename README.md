# Data Analysis Toolkit

> **Your All-in-One Workspace for Statistical Analysis & Data Science**  
> *Exploration · Descriptive Stats · Visualization · Probability Distributions · Statistical Inference*

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-3.14-green.svg)
![Scipy](https://img.shields.io/badge/SciPy-1.11+-blue.svg)
![License](https://img.shields.io/badge/LICENSE-MIT-blue.svg)

[**Quick Start**](#quick-start) · [**Features**](#features) · [**Installation**](#installation) · [**Usage**](#usage)

</div>

---

## 🎯 Overview

The **Data Analysis Toolkit** is a comprehensive Streamlit-powered application designed to streamline your entire data workflow—from initial file exploration through advanced statistical hypothesis testing and automated report generation.

Built with **Python** and powered by industry-leading libraries (**pandas, numpy, scipy, statsmodels**), this toolkit brings academic-grade statistical methods directly to your desktop environment with an intuitive, web-based interface.

---

## ✨ Key Features

### 1️⃣ **Data Exploration & Preparation**
- 📤 Smart CSV/Excel file upload with live preview
- 🔍 Missing value detection and analysis
- 🎯 Outlier identification using IQR method
- 🧹 Automated data cleaning with mean/median imputation
- 🔄 Data transformation tools (log, root, z-score)

### 2️⃣ **Descriptive Statistics**
- 📊 Central tendency measures (mean, median, mode)
- 📈 Dispersion metrics (variance, standard deviation, IQR)
- 📉 Shape analysis (skewness, kurtosis)
- 🔗 Correlation matrices with heatmaps
- 📋 Covariance matrices

### 3️⃣ **Data Visualization**
- 📉 Histograms with optional KDE smoothing
- 📦 Box plots for outlier detection
- ⚡ Scatter plots with linear trend lines
- 📊 Bar charts (grouped/stacked)
- 📈 Line charts and time series
- Q-Q plots (quantile-quantile)
- 🔥 Correlation heatmap visualizations

### 4️⃣ **Probability Distributions**
Explore six fundamental distributions with interactive PDF/CDF visualization:

| Distribution | Type | Parameters | Use Cases |
|-------------|-------|------------|-----------|
| **Normal** | Continuous | μ (mean), σ (std) | Heights, test scores |
| **Binomial** | Discrete | n (trials), p (prob) | Coin flips, success rates |
| **Poisson** | Discrete | λ (rate) | Rare events, call centers |
| **Exponential** | Continuous | λ (rate) | Waiting times, lifetimes |
| **Uniform** | Continuous | a, b (bounds) | Random sampling, lottery |
| **Bernoulli** | Discrete | p (prob) | Yes/no outcomes |

### 5️⃣ **Statistical Inference Engine**

#### Parametric Tests (Normal Distribution Assumed)
- **One-Sample t-Test** — Compare sample mean to hypothesized value
- **Two-Sample t-Test (Independent)** — Compare means of two independent groups
- **Paired t-Test** — Before/after comparisons
- **One-Way ANOVA** — Compare 3+ group means with Tukey HSD post-hoc

#### Non-Parametric Tests (No Distribution Assumptions)
- **Mann-Whitney U Test** — Alternative to two-sample t-test
- **Wilcoxon Signed-Rank Test** — Alternative to paired t-test
- **Kruskal-Wallis H Test** — Alternative to ANOVA

#### Normality Testing
- Shapiro-Wilk test
- Kolmogorov-Smirnov test
- Anderson-Darling test

### 6️⃣ **Regression Analysis**
- 📈 Simple Linear Regression (y = β₀ + β₁x)
- 🔀 Multiple Linear Regression (multiple predictors)
- 🎯 Model diagnostics: R², Adjusted R², RMSE
- 🔍 Residual analysis with Q-Q plots and residual vs fitted plots

### 7️⃣ **Confidence Intervals**
Compute confidence intervals for:
- Single sample mean (t-distribution based)
- Difference between two means (Welch's t-test)
- Population proportion (Wilson score interval)
- Mean/median/std via bootstrap resampling

### 8️⃣ **Sampling Methods**
Demonstrate fundamental sampling techniques:
- **Simple Random Sampling** — Equal probability selection
- **Stratified Sampling** — Proportional representation from each group
- **Systematic Sampling** — Fixed k-th element selection
- **Sample Size Calculator** — Required n for desired precision

### 9️⃣ **PDF Report Generation**
Automatically compile your analysis session into professional PDF reports with:
- Custom report builder with drag-and-drop ordering
- Auto-included datasets, statistics, and visualizations
- Professional formatting in Letter or A4 layout
- Export-ready documents for presentations and publications

### 🔟 **AI Assistant (Ollama)**
Chat naturally with your data using locally-hosted AI models:
```markdown
- ✅ Describe dataset and columns
- ✅ Ask statistical questions ("What's the median age?")
- ✅ Request visualizations and plots
- ✅ Run hypothesis tests via tool calls
- ✅ Save insights to persistent memory
- ✅ Add findings to your PDF report
```

**Features:**
- ⚡ Real-time chat interface
- 🧠 Tool calling for data analysis operations
- 💾 Persistent memory across sessions
- 📊 Inline visualization rendering
- 🔍 Context-aware responses with conversation summarization
- 🖥️ System usage monitoring (CPU, RAM, GPU)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (optional, for AI Assistant)
- Local machine or system with internet access

### Installation

1. **Clone or download** this repository:
```bash
cd /path/to/toolkit-app
```

2. **Activate the virtual environment**:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
python -m pip install --upgrade pip
pip install streamlit pandas numpy matplotlib scipy statsmodels
```

4. **Run the application**:
```bash
streamlit run app.py
```

5. **Access the app** at `http://localhost:8501`

---

## 📚 Features by Category

### Core Modules

| Feature | Description | Use Case |
|---------|------------|----------|
| **Data Explorer** | Upload, preview, clean, and transform datasets | Initial data inspection |
| **Descriptive Statistics** | Summary tables, correlation matrices | Understanding data structure |
| **Data Visualization** | Interactive plots and charts | Exploratory data analysis |
| **Probability Distributions** | Model and fit distributions | Predictive modeling foundation |
| **Statistical Inference** | t-tests, ANOVA, chi-square tests | Hypothesis validation |
| **Regression Analysis** | Linear regression with diagnostics | Predictive modeling |

### Advanced Tools

| Feature | Description | When to Use |
|---------|------------|-------------|
| **Confidence Intervals** | Margin of error calculation | Quantify uncertainty |
| **Non-Parametric Tests** | Distribution-free inference | Non-normal data |
| **Data Transformations** | Log, sqrt, Box-Cox, standardization | Meet test assumptions |
| **Sampling Methods** | Random, stratified, systematic sampling | Simpler experiments |

### Utilities

| Feature | Description | Output |
|---------|------------|--------|
| **PDF Report Generator** | Compile session into professional reports | PDF document |
| **AI Assistant** | Interactive chatbot for data queries | Chat interface |
| **System Monitor** | View CPU/RAM/GPU usage | Performance diagnostics |

---

## 🛠️ Installation

### Full Setup (with AI Assistant)

```bash
# Install core dependencies
pip install streamlit==1.30.0
pip install pandas numpy matplotlib scipy statsmodels

# Optional: For AI Assistant
pip install ollama psutil

# Download a model locally (requires internet):
# ollama pull llama3.2
```

### Minimal Setup (No AI)

```bash
pip install streamlit pandas numpy matplotlib scipy statsmodels
streamlit run app.py
```

---

## 📖 Usage Guide

### Getting Started

1. **Load Your Dataset**
   - Navigate to **Data Explorer** in the sidebar
   - Upload a CSV or click "Load Sample Data" for demo
   - Preview data in expanded cards (shape, types, missing values)
   - Detect outliers using IQR method

2. **Basic Analysis**
   - Run automatic summary statistics
   - Explore correlations with heatmap
   - Visualize distributions via histograms and box plots

3. **Statistical Testing**
   - Check normality assumptions: Shapiro-Wilk / Anderson-Darling
   - Based on results, choose appropriate test (parametric or non-parametric)
   - Run t-tests, ANOVA, chi-square, etc.

4. **Build Report**
   - Use "Add to Report" buttons throughout each module
   - Customize order and preview before export
   - Download professional PDF report

---

### Example Workflow

#### Business Marketing Analysis

1. **Upload Sales Dataset** → Data Explorer
2. **Explore Correlations** → Descriptive Statistics
   ```
   Revenue vs Ad Spend: r = 0.73 ***
   Revenue vs Customer_Age: r = -0.42 *
   ```
3. **Visualize Distributions** → Data Visualization
   - Histogram of spending by segment
   - Box plot comparing regions
4. **Test Hypothesis**: Premium customers spend more?
   - Check normality first (Shapiro-Wilk)
   - If normal: Two-sample t-test
   - If non-normal: Mann-Whitney U test
5. **Fit Regression Model** → Regression Analysis
   - Predict revenue from ad spend and demographics
6. **Generate Report** → PDF Export

---

### Module-by-Module Guide

#### 1. Data Explorer
```
Functionality | Command
--------------|-----------------------------------------------------------
Upload file   | Drag CSV or use file_uploader
Preview data  | Expand "Preview Data (first 50 rows)" card
Check types   | Expand "Column Data Types"
Find missing  | Expand "Missing Values"
Detect outliers | Expand "Outlier Detection (IQR method)"
Impute gaps   | Select column → Fill with Mean/Median/Mode
Export clean   | Download CSV button
```

#### 2. Descriptive Statistics
```
Functionality | Command ------------------------------|-----------------------------------------------------------
Summary table | View all computed moments by column
Deep dive     | Select any numeric column for detailed metrics
Correlation   | Matrix heatmap shows pairwise relationships
Covariance    | Covariance matrix (numerical)
Add to report | Button in each view exports result to PDF
```

#### 3. Data Visualization
```
Chart Type    | Description
--------------|-----------------------------------------------------------
Histogram     | Frequency distribution with optional KDE
Box Plot      | Quartiles, median, outliers
Scatter Plot  | Bivariate relationship with linear trend line
Bar Chart     | Grouped by category (mean values)
Line Chart    | Time series or index-based trends
Q-Q Plot      | Normality check — deviation from diagonal
Correlation   | Heatmap of pairwise Pearson correlations
```

#### 4. Statistical Inference
| Test Name                    | When to Use                          | Command |
|------------------------------|--------------------------------------|---------|
| Shapiro-Wilk                 | Check normality                     | Select column → Run test |
| One-Sample t-Test            | Compare mean to known target        | Choose column, set μ₀ |
| Two-Sample t-Test (Independent) | Compare two independent groups   | Select numeric + grouping col |
| Paired t-Test                | Pre/post or matched pairs           | Select two columns |
| One-Way ANOVA                | Compare 3+ group means              | Numeric + categorical col |
| Chi-Square Test of Independence | Categorical vs categorical       | Two categorical cols |
| Mann-Whitney U               | Non-parametric alternative to t-test | Same inputs as two-sample t-test |
| Wilcoxon Signed-Rank        | Non-parametric paired alternative    | Paired numeric columns |
| Kruskal-Wallis H            | Non-parametric ANOVA                 | Numeric + categorical col |

#### 5. Regression Analysis
```
Simple Linear Regression     → One X, one Y (y = β₀ + β₁x)
Multiple Linear Regression   → One Y, multiple Xs

Diagnostics Available:
  - R² coefficient
  - Adjusted R²
  - RMSE
  - Residual plots
  - Q-Q plot of residuals
```

#### 6. PDF Report Generator
```
Steps for PDF:
1. Collect items via "Add to Report" in other modules
2. Preview and reorder
3. Configure title/author/page size
4. Toggle auto-included sections
5. Click "Generate & Download PDF"
```

#### 7. AI Assistant
```
Chat Capabilities:
- "Describe my dataset" → describes columns
- "Show histogram of spending" → calls plot tool
- "Compare Premium vs Basic customers" → calls ttest
- "Summarize this conversation" → auto-summarizessession

Controls:
  - Model: llama3.2, mistral, or other Ollama models
  - Temperature: 0.0–2.0 (higher = more creative)
  - Tools: Enable/disable analysis tool calls
```

---

## 🧪 Supported Statistical Tests

### Parametric Tests (requires normality assumption)

| Test | From Library | Parameters |
|-------|-------------|------------|
| One-Sample t-Test | `stats.ttest_1samp` | sample, μ₀ |
| Paired t-Test | `stats.ttest_rel` | Sample1, Sample2 |
| Two-Sample t-Test (Equal Variance) | `stats.ttest_ind` | Sample1, Sample2, equal_var=True |
| Two-Sample t-Test (Unequal Variance - Welch) | `stats.ttest_ind` | Sample1, Sample2, equal_var=False |
| One-Way ANOVA | `stats.f_oneway` | Group1, Group2, ... |
| F-test for Variances | `stats.levene` | Sample1, Sample2, ... |

### Non-Parametric Tests (distribution-free)

| Test | From Library | When to Use |
|-------|-------------|-------------|
| Mann-Whitney U | `stats.mannwhitneyu` | Two independent groups, non-normal |
| Wilcoxon Signed-Rank | `stats.wilcoxon` | Paired samples, non-normal |
| Kruskal-Wallis H | `stats.kruskal` | 3+ independent groups, non-normal |

---

## 📊 Probability Distributions Reference

### Continuous Distributions

```python
# Normal: μ (mean), σ (standard deviation)
norm.pdf(x, loc=0, scale=1)
norm.cdf(x, loc=0, scale=1)

# Exponential: λ (rate parameter)
expon.pdf(x, scale=1/lambda_)
expon.cdf(x, scale=1/lambda_)

# Uniform: a (lower), b (upper)
sp_uniform.pdf(x, a=0, b=1)
```

### Discrete Distributions

```python
# Binomial: n (trials), p (probability of success)
binom.pmf(k, n, p)         # P(X = k)
binom.cdf(k, n, p)         # P(X ≤ k)

# Poisson: λ (rate parameter)
poisson.pmf(k, lambda_)     # P(X = k)
poisson.cdf(k, lambda_)     # P(X ≤ k)
```

---

## 🤖 AI Assistant Capabilities

The AI Assistant can perform the following tools on your loaded dataset:

| Tool | Description | Example Use |
|-------|-------------|-------------|
| `describe` | Compute descriptive statistics | "Describe column Age" → shows mean, std, min, max... |
| `correlation` | Pearson correlation or full matrix | "How correlated are Income and Spending?" |
| `ttest` | Two-sample t-test by group | "Compare Satisfaction_Basic vs Satisfaction_VIP" |
| `value_counts` | Frequency counts for categorical data | "Count values in Region column" |
| `missing_values` | Report missing data per column | "How many NAs are in the dataset?" |
| `plot` | Generate visualizations (hist, box, scatter, bar) | "Show histogram of Purchase_Frequency" |
| `remember` | Save facts/preferences to memory | "Remember: use log-transform for skewed data" |
| `recall` | Retrieve stored memories | "What did I save earlier?" |
| `add_to_report` | Add result to PDF report builder → "Add this to my report" |

---

## 🧠 AI Tools Architecture

The AI uses function calling with the following tool schema:

```json
{
  "tools": [
    { "name": "describe", "params": { "column": string } },
    { "name": "correlation", "params": { "col1": string, "col2": string } },
    { "name": "ttest", "params": { "column": string, "group_column": string } },
    { "name": "plot", "params": { 
        "kind": "histogram|box|scatter|bar|line",
        "column": string, "col2": string, etc.
      }
    },
    ...
  ]
}
```

---

## 🛡️ Safety Note - AI Assistant

The app integrates Ollama for local LLM inference but:
- ⚠️ Does not include safety filters (model-dependent)
- ✅ Runs entirely locally — no data sent externally
- 🔒 Always validate outputs before applying to production

**Recommendation**: Verify statistical outputs manually for critical analyses.

---

## 🎨 UI Tour

```
┌──────────────────────────────────────────────────────────────┐
│  Data Analysis Toolkit                             ⚙ Menu   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sidebar Navigation:                                         │
│    • Data Explorer         → Upload & clean datasets        │
│    • Descriptive Stats     → Summary tables & correlations  │
│    • Data Visualization    → 7 chart types                  │
│    • Probability Distributions  → Fit models                │
│    • Statistical Inference  → t-tests, ANOVA, Chi-square   │
│    • Regression Analysis    → Linear regression diagnostics │
│    • Confidence Intervals   → CIs for means/proportions    │
│    • Non-Parametric Tests   → Mann-Whitney / Kruskal       │
│    • Transformations        → Log / sqrt / z-score         │
│    • Sampling Methods       → Random, stratified sampling   │
│    • PDF Report Generation  → Professional report export    │
│    • AI Assistant          → Chat with your data           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
toolkit-app/
├─ app.py                    # Main Streamlit application
├─ src/
│  ├─ ai_assistant.py        # AI integration & tool definitions
│  ├─ data_explorer.py       # File upload, preview, cleaning
│  ├─ distributions.py       # Probability distribution module
│  ├─ home.py               # Landing page (welcome/dashboard)
│  ├─ inference.py          # Hypothesis testing module
│  ├─ statistics.py         # Descriptive statistics engine
│  ├─ transformations.py     # Data transforms (log, z-score, etc.)
│  ├─ visualization.py      # Plotting and chart generation
│  └─ sampling.py           # Sampling methods demo
├─ venv/                     # Virtual environment
└─ README.md                 # This file
```

---

## 🔧 Configuration Options

### Streamlit Config (`~/.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
port = 8501
address = '0.0.0.0'
headless = true
```

### Ollama Environment Variables (`.env`)

```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama not responding** | Ensure `ollama serve` is running in background; check model loaded (`ollama list`) |
| **ImportError: ollama not found** | Install: `pip install ollama` |
| **Dataset upload fails** | Check CSV format (UTF-8 encoding); ensure comma separators |
| **t-test produces error** | Verify numeric column and ≥2 categories in grouping variable |
| **Plot too large / slow** | Reduce dataset size or adjust point density manually |

---

## 🔨 Dependencies

### Core (Required)

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib` | Static plotting |
| `scipy` | Statistical functions & tests |
| `statsmodels` | Advanced statistical modeling |

### Optional

| Package | Feature |
|---------|---------|
| `ollama-python` | Local LLM inference |
| `psutil` | System resource monitoring |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Credits

**Built for SRH · Tools & Methods of Data Analysis**

- **Core Libraries**: pandas, numpy, scipy, statsmodels
- **Frontend Framework**: Streamlit
- **Visualization**: Matplotlib
- **PDF Generation**: ReportLab
- **AI Integration**: Ollama API via `ollama-python`

---

## 📞 Support & Contributing

This is an educational tool for learning statistical methods. For questions about:
- Statistical methodologies → Consult academic resources
- App bugs/feature requests → Report issues in repository
  
> "Statistics without data analysis is like sailing without a map." — Anonymous

---

**Version**: 1.0.0  
**Last Updated**: June 2026  
Made with ❤️ using Python
