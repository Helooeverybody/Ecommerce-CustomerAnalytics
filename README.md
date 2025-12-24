# Ecommerce-CustomerAnalytics

## Projects description
Customer growth and retentions has been central challenges for data-driven 
organization seeking to optimize marketing efficiently and long-term profitability, 
regardless of being in the era of unprecedented AI advancements. The project proposes
an integrated analytic pipeline combining Customer Segmentation, Predictive CLV, Churn
Prediction and Experiment based on Policy assumptions to generate actionable insights 
and personalized interventions. First, segmentation techniques are applied to identify 
the heterogeneous customer groups based on customer behaviors, thus predictive CLV 
models estimate the future contribution of each customer from each group. Churn prediction 
module is then utilized to detect at-risk users, finally, in decision making, Monte 
Carlo experiment quantify the incremental impact of assumption-based policy simulation, 
allowing the design of targeted strategies that maximize conversion and retention lift. 
Together, this end-to-end pipeline provide robust framework for the problem of customer growth.

Data link: [raw-data](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset/data)
## Pipeline Overview
<img width="1128" height="364" alt="image" src="https://github.com/user-attachments/assets/c4ab04e4-8c14-4fc4-ba17-f517926b0745" />

## Folder structures

```
.
├── data/
|   ├── raw/                         # Raw Data
|   ├── processed/                   # Processed Data
├── notebooks/
│   ├── 1_eda.ipynb                  # Exploratory Data Analysis (EDA)
│   ├── 2_preprocess.ipynb           # Data cleaning & feature engineering
│   ├── 3_segmentation.ipynb         # Customer segmentation (e.g. GMM)
│   ├── 4_clv.ipynb                  # Customer Lifetime Value modeling
│   ├── 5_churn.ipynb                # Churn prediction models
│   ├── 6_experiment.ipynb           # Policy simulation & experiment design
│   └── 7_dashboard.ipynb            # Dashboard & business insights
│
├── src/
│   ├── preprocess.py                # Preprocessing logic (reusable pipeline)
│   ├── segmentation.py              # Segmentation models & utilities
│   ├── clv.py                       # CLV computation & prediction
│   ├── churn.py                     # Churn modeling logic
│   └── exp.py                       # Decision policies & experiment simulation
│
├── output/
│   ├── figures/                     # Saved plots & visualizations
│   └── abnormal_detection_output/   # Serialized models (if any)
│
└── requirements.txt                 # Project dependencies

```

## Dashboard
### Operational Dashboard
<img width="1313" height="733" alt="image" src="https://github.com/user-attachments/assets/59c8c0bf-0078-4879-9cc1-5932937ca96d" />
### Strategic Dashboard:
<img width="1308" height="734" alt="image" src="https://github.com/user-attachments/assets/448de72f-8f0d-498a-8633-b268f4861c5f" />
