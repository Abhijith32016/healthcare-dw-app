Healthcare Data Warehouse — Disease Surveillance & Outbreak Prediction

A cloud-deployed healthcare analytics platform that ingests national-scale epidemiological data through a distributed big-data pipeline, flags statistical outbreak anomalies, forecasts case trajectories with three independent model families, fuses OCR-extracted signal from handwritten clinical documents, converts predictions into operational decisions via linear-programming budget optimisation, and serves results through both a Power BI executive dashboard and a live public web application.

Live app: healthcare-outbreak-dw.streamlit.app

Pipeline Overview
Raw Epidemiological CSVs (185K+ rows COVID-19 + 120K+ rows US Infectious Disease)
        │
        ▼

│  PySpark ETL         │  Distributed windowed aggregation, rolling 7/14-day
│  (Google Colab)      │  averages, lag features, z-score anomaly detection

          │  Partitioned Parquet
          ▼

│  Azure Data Lake     │  Hierarchical namespace storage (ADLS Gen2)
│  Storage Gen2        │  Lakehouse pattern — external views over Parquet via Azure Synapse serverless SQL (OPENROWSET)
          │
          ▼

│  ML & Decision Layer │  ARIMA forecasting · Random Forest · XGBoost
│                      │  TrOCR clinical-note fusion · PuLP LP optimiser

          │
          ▼

│  Delivery Layer      │  Power BI executive dashboard (DAX measures)
│                      │  Streamlit live web app (interactive prediction)

What This Project Demonstrates
Big data engineering: Distributed ETL with PySpark window functions across 300K+ combined rows — rolling aggregations, lag features, and z-score outbreak detection computed at scale, output as partitioned Parquet (not CSV).
Cloud data warehousing: Designed, deployed, and adapted a cloud warehouse architecture mid-build when a genuine platform constraint was discovered (Azure Synapse serverless pools don't support CREATE TABLE / INSERT INTO) — resolved by redesigning as an external-view lakehouse over Data Lake Parquet via OPENROWSET. The resulting architecture is more cost-efficient than the original plan.
Applied machine learning across three paradigms:
Classical time-series: ARIMA with stationarity testing (Augmented Dickey-Fuller) driving differencing order
Ensemble classification: Random Forest (ROC-AUC 0.981) and XGBoost (ROC-AUC 0.993) trained on 27 engineered features, with class-imbalance handling (scale_pos_weight, class_weight="balanced") as a first-class modeling decision
Applied pretrained deep learning: TrOCR (Hugging Face) for handwriting recognition on medical prescriptions, with RapidFuzz fuzzy matching for drug-name extraction — 50% exact-match accuracy reported honestly as a real, explained OCR limitation on domain-specific vocabulary
Prescriptive analytics / operations research: Linear programming (PuLP, CBC solver) for constrained budget allocation — maximise risk-weighted outbreak-response coverage across regions subject to total-budget and per-region caps. An early unconstrained iteration exposed a degenerate solution (100% of budget to one statistical outlier region); the constraint addition and re-validation is documented.
Multi-modal data fusion: Structured epidemiological surveillance signal combined with unstructured clinical-text signal (OCR-extracted handwritten prescriptions) — a current direction in public-health informatics.
Production debugging discipline: Seven documented deployment failures across four cloud/platform boundaries (Colab, Azure, GitHub, Streamlit Cloud), each root-caused via log inspection rather than guesswork — not a tutorial walkthrough.
End-to-end ownership: From raw Kaggle download through a live, publicly reachable, dynamically working application — not a notebook that only runs on one machine.
Core Components
1. Big Data ETL (PySpark)
Processes 185,000+ rows of COVID-19 time-series data and 120,000+ rows of US infectious disease records (300K+ combined rows) using PySpark distributed windowed aggregation
Rolling 7-day and 14-day averages, lag features, and momentum/acceleration features computed via window functions partitioned by country
Z-score anomaly detection (threshold > 2.0) produces a severity-tiered alert system: low / medium / high / critical
Output written as partitioned Parquet — not CSV — to Azure Data Lake Storage Gen2
2. Cloud Data Warehouse (Azure Synapse Analytics — lakehouse pattern)
Originally designed as a Kimball star schema with dedicated fact/dimension tables
Mid-build discovery: Synapse serverless SQL pools don't support CREATE TABLE / INSERT INTO — a genuine architectural constraint, not a config error
Redesigned as external views over Parquet files via OPENROWSET — a lakehouse pattern that trades minor query-time cost for zero idle compute cost
End-to-end verified: SELECT COUNT(*) against Synapse views returned exactly the row count the Spark job produced, confirming no data loss across the ETL → Data Lake → Synapse boundary
3. Machine Learning & Forecasting Layer
ARIMA: Per-country time-series forecasting with Augmented Dickey-Fuller stationarity testing determining differencing order
Random Forest: Binary outbreak classifier, 27-feature engineered set, ROC-AUC 0.981, full cross-validation and classification report on imbalanced target
XGBoost: Same task, ROC-AUC 0.993, scale_pos_weight for class imbalance, cross-validated
TrOCR + RapidFuzz: Handwriting recognition on Kaggle medical prescription dataset; 50% exact-match accuracy on drug-name extraction — reported honestly with root-cause explanation (general-purpose OCR on domain-specific vocabulary), not inflated
4. Decision Support Layer (Prescriptive Analytics)
Capacity planning: ARIMA forecasts translated into projected hospital bed-demand using hospitalization-rate / length-of-stay model, flagging projected shortfall days
Budget allocation (LP): PuLP constrained optimisation — maximise risk-weighted coverage of outbreak-response budget, subject to total-budget constraint and per-region cap. Degenerate-solution failure mode caught, diagnosed, and fixed via constraint addition
Cost-of-delay modelling: Exponential-growth cost curve quantifying economic cost of delayed intervention — labelled explicitly as an illustrative model, not a validated health-economics figure
5. Delivery Layer
Power BI: Two-page executive dashboard (surveillance overview; business decision support) with custom DAX measure Budget Allocation Efficiency (risk-covered per dollar)
Streamlit: Live public web app — surveillance dashboard, business decision-support views, and interactive outbreak-prediction tool backed by actual trained RF/XGBoost models
Technology Stack
Layer	Technology	Why
Big data processing	PySpark (distributed, windowed)	Native partitioned time-series aggregation at scale
Data lake / storage	Azure Data Lake Storage Gen2	Hierarchical namespace, native Synapse integration
Data warehouse	Azure Synapse Analytics (serverless SQL, lakehouse pattern)	Query-time cost model over Parquet, zero idle compute
Time-series forecasting	statsmodels ARIMA	Classical, interpretable forecasting with stationarity testing
Classification	scikit-learn Random Forest, XGBoost	Two independent model families for cross-validated comparison
OCR / handwriting	Hugging Face Transformers (TrOCR, pretrained)	State-of-the-art handwriting recognition without training-from-scratch cost
Fuzzy text matching	RapidFuzz	Robust entity matching against noisy OCR output
Optimisation	PuLP (linear programming, CBC solver)	Constrained resource allocation — prescriptive analytics
BI / reporting	Power BI Desktop, DAX	Non-technical stakeholder dashboarding
Deployment	Streamlit Community Cloud	Free, public, git-integrated continuous deployment
Version control	Git + GitHub	Full history, large-file handling via CLI push
Deployment & Engineering Challenges Solved
Challenge	Root Cause	Resolution
PySpark session failed to start (JAVA_GATEWAY_EXITED)	JAVA_HOME pointed at mismatched JDK version vs system default	Diagnosed via JVM inspection; repointed to correct installed JDK path
Synapse CREATE TABLE / INSERT INTO failed on every table	Serverless SQL pools don't support persisted tables — genuine architectural constraint	Redesigned warehouse layer as external views over Data Lake Parquet via OPENROWSET — lakehouse pattern
Azure resource creation blocked by region policy	Azure for Students subscriptions carry randomised, account-specific region allowlists	Queried the actual policy assignment; redeployed in a supported region
Budget-allocation LP put 100% of funds into one tiny-population region	Unconstrained decision variables let solver exploit statistically noisy z-scores from low-case-count regions	Added per-region allocation cap and minimum case-volume filter; re-validated result
Bulk file upload to Data Lake silently overwrote data across partitions	Identical Spark part-filenames across different month=X folders collided when flattened by browser upload	Switched to Azure Storage Explorer for structure-preserving bulk upload
Streamlit Cloud deployment failed to build	Pinned pillow==10.4.0 had no prebuilt wheel for platform's newer Python runtime	Removed hard version pins from requirements.txt; let resolver select available prebuilt wheels
GitHub browser upload rejected trained model file	GitHub browser UI enforces 25MB per-file cap, well below actual 100MB Git limit	Switched to git CLI push, which handles the ~38MB model file within Git's real limits
Repository Structure
healthcare-dw/
├── app.py                        # Streamlit app — dashboard, decision support, prediction tool
├── requirements.txt              # Python dependencies for Streamlit Cloud
├── data/
│   └── *.csv                     # Pre-aggregated outputs exported from full pipeline
│                                 # (full raw/processed data lives in Azure / Google Drive)
├── models/
│   ├── rf_outbreak_model.pkl     # Trained Random Forest classifier
│   └── xgb_outbreak_model.json  # Trained XGBoost classifier
└── README.md
Run Locally
bash
pip install -r requirements.txt
streamlit run app.py
Possible Future Work
Fine-tune TrOCR on medical-vocabulary handwriting to close the gap between the current 50% and what a domain-adapted model could achieve
Extend the LP budget allocator to a multi-period rolling-horizon formulation rather than a single static allocation
Add formal backtesting of ARIMA forecasts against held-out periods with MAPE/RMSE metrics
Load-test the Streamlit deployment and Synapse serverless queries under concurrent multi-user access
Migrate the warehouse layer to a dedicated Synapse SQL pool if always-on compute becomes cost-justified
