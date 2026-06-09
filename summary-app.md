# 📊 Data Analysis Toolkit — Technical System Documentation

old app 

## 🛠️ Global Dependency & Integration Flags

The toolkit monitors system architecture and environmental bindings dynamically using the following flags:

* **`_HAS_PSUTIL` (`bool`):** Tracks if the `psutil` library is accessible. Controls local OS resource monitoring (CPU, RAM virtualization parsing).
* **`_HAS_OLLAMA` (`bool`):** Tracks if the local `ollama` library bindings are initialized. Controls native local LLM token parsing, model listings, and chat stream inferences.
* **System Platform Detection (`platform.system()`):** Tracks host operating runtime conditions.
    * `"Darwin"`: Fallbacks to system control strings (`machdep.cpu.brand_string`) to track Apple Silicon hardware.
    * `"Linux"` / `"Windows"`: Spawns sub-process calls directly to `nvidia-smi` to monitor GPU usage statistics.

---

## 📂 Core UI Architecture & Navigation Routing Map


State tracking and component flow rely entirely on the Streamlit session state tracking keys: `st.session_state["df"]`, `st.session_state["ai_messages"]`, `st.session_state["report_items"]`, and `st.session_state["ai_memory"]`. 

The application uses a categorical structural layout driven by the sidebar variable `tool`. The interface maps routing states down across the following directory:

1.  **`Home`**: Overview dashboard outlining structural frameworks, tabbed industry workflows, and automated workspace validation banners.
2.  **`Data Explorer`**: Feature matrices previewer combined with mean/median missing cell imputations and drop row cleanings.
3.  **`Descriptive Statistics`**: Extended summary metrics mapping central tendencies, variances, skewness, kurtosis, and report append routines.
4.  **`Data Visualization`**: High-resolution rendering station for Histograms, Boxplots, and multi-variable Correlation Heatmaps.
5.  **`Transformations`**: Scale manipulation tools supporting Log ($ln(x)$), Square Root, and Min-Max scaling engines.
6.  **`Probability Distributions`**: Probability modeling sandbox supporting Continuous (Normal, Exponential) and Discrete (Binomial, Poisson) evaluation metrics.
7.  **`Hypothesis Testing`**: Advanced inference engine running Shapiro-Wilk and Kolmogorov-Smirnov checking, Independent Welch t-tests, and One-Way ANOVA with Tukey HSD Post-hoc tables.
8.  **`Regression Analysis`**: Predictive multi-variable linear fitting engines calculating metrics like $R^2$ and Mean Squared Error (MSE).
9.  **`Sampling & Simulation`**: Random subset generation frame engine mapping custom matrix selections.
10. **`AI Assistant`**: Interactive local LLM Chat system with conversational memory hooks, custom system prompting, and automated contextual dataframe summary injection.
11. **`Generate PDF Report`**: Publication-ready artifact generator featuring reportlab paragraph flow control structures and custom tabular rendering.

---

## ⚙️ Function Reference & Footprints

### 1. Mathematical & Statistical Calculators

#### `normality_table(series: pd.Series) -> pd.DataFrame`
* **Description:** Computes three separate statistical normality assessment models on a data sequence to evaluate distribution criteria.
* **Logic Rules:** * If sample size $N < 3$, returns an insufficient data flag.
    * If $3 \le N \le 5000$, calculates the **Shapiro-Wilk** test statistics.
    * If $N > 5000$, skips the Shapiro-Wilk check safely to avoid SciPy computation limits.
    * Always executes a normalized **Kolmogorov-Smirnov (KS)** test alongside an **Anderson-Darling** critical value evaluation.

#### `p_stars(p_value: float, num: int = 10) -> str`
* **Description:** Converts raw statistical probability outputs ($p$-values) into symbolic structural significance stars for easier analytical output reading.

---

### 2. Layout & Graphics Formatters

#### `nice_fig(figsize: tuple = (8, 6)) -> (plt.Figure, plt.Axes)`
* **Description:** Instantiates an isolated matplotlib layout tracking tight alignments with consistent layout scales, avoiding canvas bleedover during execution cycles.

---

### 3. PDF Generator Helpers

#### `make_pdf_safe_table_data(dataframe: pd.DataFrame, style: ParagraphStyle) -> list`
* **Description:** Iterates down a Pandas DataFrame, cleans `NaN` nodes, explicitly converts values to alphanumeric strings, and wraps every cell in an isolated ReportLab `Paragraph` container. 
* **Why it's important:** Prevents clipping and out-of-boundary alignment failure crashes in ReportLab tables.

---

### 4. AI Engine & Infrastructure Monitors

#### `get_ollama_models() -> list`
* **Description:** Dispatches a query request down to the local Ollama daemon service via API, parsing model strings to populate the UI dropdown box. Returns an empty array if the system is unreachable.

#### `get_system_stats() -> dict`
* **Description:** Checks CPU utilization metrics, unified RAM allocation balances, and GPU performance thresholds via OS sub-process hooks.

#### `get_dataframe_context(df: pd.DataFrame) -> str`
* **Description:** Compiles dataset shapes, counts missing records, breaks down data-type splits, and includes numerical matrix profiles.
* **Token Constraint:** Automatically clips string lengths at a strict **4000-character ceiling** to prevent prompt buffer overflows in local LLM context windows.

#### `estimate_tokens(text: str) -> int`
* **Description:** Estimates message token size by applying a standard operational token character metric divisor ($\approx \text{length} / 4$).

#### `estimate_messages_tokens(messages: list) -> int`
* **Description:** Processes a collection of multi-turn chat records to monitor usage limits against the model's native context block size.

#### `summarize_conversation(model: str, messages: list, options: dict) -> str`
* **Description:** Automatically compiles historical conversation logs into short bullet points when context limits approach critical thresholds.

---

### 5. Persistent AI Memory & Persistent Storage CRUD

#### `load_ai_memory() -> list`
* **Description:** Reads a local file called `ai_memory.json` to load previously saved analytical insights back into session memory upon app launch.

#### `save_ai_memory(memory_list: list) -> bool`
* **Description:** Encodes and saves systemic user takeaways and data observations as a structured array back to disk storage.

---

### 6. AI Agent Tool Footprints

The application maps specific functions to the local LLM agent loop using custom tool schema definitions (`AI_TOOL_SCHEMA`):

* **`ai_tool_describe(df, column=None)`**: Provides a statistical breakdown (`.describe()`) or values frequency string for specific vector arrays.
* **`ai_tool_correlation(df, col1=None, col2=None)`**: Computes Pearson correlations for a pair of features or outputs the complete correlation dataframe.
* **`ai_tool_ttest(df, column, group_column)`**: Splits continuous metrics by a categorical factor, runs a Welch independent two-sample test, and reports structural significance.
* **`ai_tool_value_counts(df, column)`**: Returns sorted top-20 frequency rows for categorical evaluation.
* **`ai_tool_missing(df)`**: Identifies and isolates columns containing empty cell nodes.
* **`ai_tool_remember(content, category="note")`**: Commits specific text notes directly into persistent memory storage.
* **`ai_tool_recall(query=None)`**: Filters saved notes based on a user keyword query.
* **`ai_tool_add_to_report(df, title, content)`**: Automatically appends custom observation paragraphs into the executive summary stack inside `st.session_state.report_items`.
* **`ai_tool_plot(df, kind, column=None, col2=None, group_column=None, bins=30)`**: Securely handles matplotlib plotting instructions, encodes the generated canvas into raw PNG bytes, caches the image buffer, and drops the asset into the active chat visual pane.

---

## 🔄 Technical Component Dependencies
┌──────────────────┐
                 │ Sidebar Uploader │
                 └────────┬─────────┘
                          │
         Loads DataFrame (CSV/XLSX) or Demo
                          │
                          ▼
               ┌────────────────────┐
               │ st.session_state   │◄───(State Engine)
               └──────────┬─────────┘
                          │
 ┌────────────────────────┼────────────────────────┐
 ▼                        ▼                        ▼
┌──────────┐            ┌──────────┐             ┌───────────┐
│ Analytics│            │ AI Agent │             │ ReportLab │
│ Modules  │            │ (Ollama) │             │ PDF Engine│
└────┬─────┘            └────┬─────┘             └─────┬─────┘
│                       │                         │
│ (Adds Dataframes)     │ (Adds Notes)            │
└───────────────┬───────┴─────────────────────────┘
│
▼
┌───────────────────────────┐
│ report_items Staging Area │
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│   Generated PDF Output    │
└───────────────────────────┘