# Cybersecurity Threat Intelligence Project (ML + RAG)

**Target Audience**: Autonomous AI Agents / LLMs executing automated code analysis, technical report generation, and system mappings.
**Readability Optimization**: Structural, high-density factual payload, strictly avoiding Markdown bloat.

## 1. System Architecture Overview
The composite threat classification system integrates tabular machine learning, deep learning classifiers, and generative knowledge retrieval into a unified Streamlit execution context (observable in `model_training.py` and `complete_project.py`).
1. **Machine Learning Incident Predictor**: Leverages synthetically generated organizational attack geometries (financial loss, affected users, MTTR). Tabular data undergoes preprocessing (LabelEncoder, StandardScaler) and is passed through classical classifiers (Logistic Regression, Random Forest, Gradient Boosting) alongside a Keras Deep Neural Network (`incident_dl_model.h5`) for categorical binary classification of organizational vulnerability thresholds.
2. **Deep Learning URL Checker**: Distinct execution boundary strictly evaluating raw URL semantics. Ingests uniform resource locators, maps syntax to numeric feature tensors (regex filters matching special chars, IP-based domains, specific target keywords), and utilizes either a specialized RF estimator or a sequential dense NN (`url_dl_model.h5`) to extrapolate phishing/malware confidence probabilities.
3. **LangChain/FAISS RAG Engine**: Retrieval-Augmented Generation module embedding unstructured technical definitions from `knowledge.txt` via `sentence-transformers/all-MiniLM-L6-v2`. The resultant vector topology is indexed locally via FAISS. Generation is orchestrated through either a specialized local HF pipeline (`distilgpt2` inside `rag_engine.py`) or a grounded Gemini 2.5 REST API implementation natively executing within the Streamlit lifecycle loop.

## 2. Repository Map
* `comparison_analysis.py` -> Evaluates JSON performance payloads (`training_results.json`) to plot AUC/ROC logic paths and extrapolate comparative classifier efficacy via `matplotlib` and `seaborn`. Outputs `comparison_report.json`.
* `complete_project.py` -> Unified project sandbox. Orchestrates real-time synthetic data generation, Random Forest parameter fitting, semantic URL vectorization, and a monolithic Streamlit UI (Incident + URL Checker + RAG).
* `data_collection.py` -> Pre-runtime data generator class (`CybersecurityDataCollector`). Outputs `cybersecurity_threats.csv` populated via normal/exponential distributions simulating incident volumes.
* `data_preprocessing.py` -> Feature engineering pipeline (`loss_per_user`, `efficiency_ratio`, `risk_score`). Selects top matrices via `SelectKBest` and exports transformation objects.
* `deployment.py` -> Streamlit consumption layer. Binds `.pkl` artifacts (`baseline`, `random_forest`, `gradient_boosting`) directly to reactive React/Plotly components.
* `knowledge.txt` -> Unstructured payload of base-truth facts explicitly defining ransomware/DDoS bounds. Ingested by FAISS.
* `model_training.py` -> The comprehensive architecture script. Computes multi-estimator evaluations, fits Keras `Sequential` architectures for both Incident (`incident_dl_model.h5`) and URL evaluation (`url_dl_model.h5`), serializes state binaries, and exposes a Streamlit frontend containing a grounded Gemini implementation loop.
* `rag_engine.py` -> Handles semantic vector encoding logic. Initializes `RecursiveCharacterTextSplitter`, local FAISS topology, and the `distilgpt2` pipeline. Exposes `ask_rag(vector_db, question)`.
* `requirements.txt` -> Environment topological lockfile.
* `test_rag.py` -> Invokes `knowledge.txt` vector search and contextually triggers the LLM pipeline locally as a distinct module integrity check.
* `train_url_model.py` -> Specialized runtime isolating Random Forest induction specifically on `extract_url_features` tensor derivatives. Emits `url_model.pkl` and `url_scaler.pkl`.

## 3. Tech Stack & Libraries
* **Core ML & DL**: `scikit-learn` (RandomForest, GradientBoosting, LogisticRegression), `tensorflow` & `keras` (Sequential NN).
* **NLP & Vectorization**: `transformers` (`distilgpt2` pipeline), `langchain-community`, `langchain-core`, `langchain-text-splitters`, `sentence-transformers`, `faiss-cpu`.
* **External APIs**: `requests` (Gemini grounding payload API abstraction).
* **Data Processing**: `pandas`, `numpy`, `urllib.parse`, `re` (Regex primitives).
* **Dashboarding & Viz**: `streamlit`, `plotly.express`, `plotly.graph_objects`, `matplotlib`, `seaborn`.

## 4. Data Dictionary
### Table 4.1: Incident Matrix (Continuous & Discrete Features)
* `financial_loss`: [Continuous] Exponential distribution indicating immediate financial impact.
* `affected_users`: [Discrete] Poisson distribution of compromised endpoints/users.
* `response_time`: [Continuous] Gamma-distributed mean time to contain/respond (MTTR).
* `vulnerability_score`: [Continuous] Uniform baseline coefficient measuring defense topology.
* `data_breach_size`: [Continuous] Log-normal mapping of payload extraction in MB.
* `network_traffic`: [Continuous] Gaussian-distributed gigabyte flow variables assessing DDoS thresholds.
* `loss_per_user` / `efficiency_ratio` / `risk_score`: [Synthesized] Engineered fractional indices bridging base parameters.
* `attack_type` / `target_industry` / `country`: [Categorical] `LabelEncoder` outputs mapping explicit taxonomy string descriptors.
* `high_risk_incident`: [Binary/Target] Aggregated risk proxy condition (Loss>80 || Users>30000 || RT>8).

### Table 4.2: URL Vector Representation Features
* `url_length` / `domain_length`: [Integer] Character width bounding limits.
* `num_digits` / `num_special_chars` / `num_dots` / `num_hyphens` / `num_subdirs`: [Integer] Frequency counts of lexical structures often obfuscated via algorithmic generation or DGA bots.
* `has_https`: [Binary] Protocol encryption enforcement boolean.
* `has_ip`: [Binary] Regex match bounding generic IPv4 structures embedded dynamically inside the `netloc`.
* `has_at`: [Binary] Boolean detection of `@` credentials chaining obfuscation strategies.
* `suspicious_words`: [Integer] Vectorization matching the presence array for ["login", "verify", "secure", "update", "bank"].
## 5. Execution & Runtime Topology
* **Environment Instantiation**: 
  `python -m venv venv` 
  `.\venv\Scripts\activate` (Windows)
* **Dependency Resolution**: 
  `pip install -r requirements.txt`
* **Artifact Generation (Pre-Requisite)**: 
  `python model_training.py` -> Must be executed in bare Python first to serialize `.pkl` and `.h5` model binaries to the local directory.
* **Primary Dashboard Entry Point**: 
  `streamlit run model_training.py` -> Initializes the React frontend on `localhost:8501`, binds the models, and opens the REST pipeline for the Gemini RAG bot.
* **API Constraints**: The Streamlit runtime requires a valid Google AI Studio API key injected into `API_URL` within `model_training.py` to bypass `403/404` REST errors during RAG execution.