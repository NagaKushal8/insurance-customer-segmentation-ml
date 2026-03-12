# Insurance Customer Segmentation

## Overview

Insurance customers aren't all the same , they differ in age, income, risk, and what they actually need. This project analyzes 10,000+ real customer records and uses machine learning to automatically group them into 6 distinct profiles. The result is a clear, data-backed picture of who the customers are, with actions like:

* Cross-sell premium health plans to high-income retirees
* Pull back marketing spend on high-loss young adult segments
* Target middle-aged families specifically for motor insurance

---

## Problem Statement

Insurance companies serve highly heterogeneous customer bases, yet often apply one-size-fits-all strategies for pricing, product bundling, and customer retention. Without a structured segmentation framework, high-value customers are undertreated, high-risk customers are under-priced, and marketing budgets are spread inefficiently across the portfolio.

The goal is to identify distinct customer groups with varying product interests, market participation, and responsiveness to marketing — then use those insights to optimize decisions around promotions, pricing, and targeted outreach.

---

## Dataset

`insurance_dataset.csv` — **10,290 rows × 13 columns**, with approximately **2% missing values**.

Premiums reflect annual figures for 2016. Negative premiums may appear where policy reversals in the current year were paid in a prior year.

| Variable | Description |
|---|---|
| `ID` | Customer ID |
| `First Policy` | Year of the customer's first policy |
| `Birthday` | Customer's birth year |
| `Education` | Academic degree (Basic / High School / BSc/MSc / PhD) |
| `Salary` | Gross monthly salary (€) |
| `Area` | Geographic living area |
| `Children` | Binary — has children (1 = Yes) |
| `CMV` | Customer Monetary Value |
| `Claims` | Claims rate |
| `Motor` | Motor insurance premium (€) |
| `Household` | Household insurance premium (€) |
| `Health` | Health insurance premium (€) |
| `Life` | Life insurance premium (€) |
| `Work Compensation` | Work compensation insurance premium (€) |

> **Customer Lifetime Value** = (Annual Profit from the Customer) × (Number of Years as a Customer) − (Acquisition Cost)

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `yellowbrick` |
| Preprocessing | `scikit-learn` (`SimpleImputer`, `StandardScaler`, `MinMaxScaler`) |
| Encoding | `category_encoders` |
| Clustering | `scikit-learn` (`KMeans`, `AgglomerativeClustering`, `GaussianMixture`) |
| Hierarchy analysis | `scipy` (`dendrogram`) |
| Evaluation | `silhouette_score`, `silhouette_samples`, `normalized_mutual_info_score` |
| Hyperparameter tuning | `GridSearchCV` |

---

## Project Pipeline

```
Raw Data
   │
   ▼
1. Data Preprocessing (Data Munging)
   ├── Null value check (~400 missing values)
   ├── Duplicate value check
   ├── Data type check
   ├── Coherence checks:
   │     ├── Removed invalid birth years
   │     ├── Fixed records where First Policy year < Birth year
   │     └── Removed records where total premiums > annual salary
   └── Missing value imputation:
         ├── Continuous variables → mean
         └── Categorical variables → mode
   │
   ▼
2. Exploratory Data Analysis (EDA)
   ├── Premium distribution by education degree
   ├── Age vs. median premiums by insurance type
   └── Salary vs. total premium correlations
   │
   ▼
3. Feature Engineering
   ├── total_premium     = Σ all premium types
   ├── commitment        = annual_salary / total_premium
   ├── profit_percent    = ((total_premiums − claim_amount) / claim_amount) × 100
   ├── retention_cost    = (profit × no. of years) − CMV
   ├── loss              = 0 if no loss occurred, 1 if loss occurred
   ├── Age               = derived from Birthday
   ├── Education encoded as numeric (1=Basic, 2=High School, 3=BSc/MSc, 4=PhD)
   └── Salary binned into six income categories
   │
   ▼
4. Outlier Removal
   ├── IQR method
   └── Manual domain-specific thresholding
       └── Treated columns: all_premiums, monthly_salary, CMV, claim_rate
   │
   ▼
5. Scaling
   └── MinMaxScaler (selected after comparing multiple scaling techniques)
   │
   ▼
6. Clustering — three algorithms evaluated
   ├── K-Means          (Elbow method → optimal k = 6, distortion score = 5332.356)
   ├── Agglomerative    (Ward linkage + dendrogram; best silhouette score ~0.41 at k=5–6)
   └── Gaussian Mixture (BIC/AIC model selection; silhouette scores evaluated across k=2–9)
   │
   ▼
7. Model Evaluation & Selection
   ├── Silhouette Score (primary metric)
   ├── Normalized Mutual Information
   └── GridSearchCV for parameter fine-tuning on the best-performing algorithm
   │
   ▼
8. Post-Clustering EDA & Segment Profiling
   ├── Median profit % per cluster
   ├── Median commitment & age per cluster
   ├── Median CMV vs. commitment per cluster
   └── Age vs. income per cluster
```

---

## Results

### Clustering Model Selection

Three algorithms were evaluated — K-Means, Agglomerative (Hierarchical), and Gaussian Mixture Models — using silhouette scores as the primary metric. The elbow method identified **k = 6** as the optimal number of clusters (distortion score = 5,332). Agglomerative clustering achieved its highest silhouette score (~0.41) at k = 5–6, consistent with K-Means. All algorithms produced structurally similar segmentations, confirming cluster stability.

### Discovered Customer Segments

Six distinct customer profiles were identified:

| Cluster | Segment Name | Key Characteristics |
|---|---|---|
| 0 | **Profitable Mid-life Customers** | Salary €3k–4k/mo · Educated · Most have children · Lowest claim rate · Largest retention cost · Second-highest motor premiums · Second-lowest health premiums |
| 1 | **Educated Young Adults (Low Income)** | Born ~1985 · Salary €1k–2k/mo · Educated · High premiums relative to income · Low retention cost · Overall low-risk, secure clients |
| 2 | **Affluent Seniors (No Children)** | Avg. age 69 · Salary €3k–4k/mo · BSc/MSc educated · No children · Low claim rate · Heaviest spend on health premium |
| 3 | **Middle-aged Families (Mid Income)** | Avg. age 48 · Salary €2k–3k/mo · Educated · At least 1 child · Highest motor premiums · Lower spend on life, work, and health |
| 4 | **High-net-worth Oldest Customers** | Salary >€4k/mo · Educated · Mostly no children · Highest health premium · Highest commitment required · Lowest loss % |
| 5 | **Young Adults (Very Low Income)** | Max income ~€1k/mo · Mostly basic education · At least 1 child · Avg. age ~22 · Highest claim rate · Highest household premiums · Lowest profit %, highest loss % |

---

## Business Strategies

Based on the segment profiles, the following actions are recommended:

1. **Focus on middle-aged customers (Cluster 0 & 3)** for standard insurance plans — reliable payers with established household and motor needs.
2. **Target young adults (Clusters 1 & 5)** with basic, entry-level insurance plans to build long-term relationships early.
3. **Tailor premium health and life offerings for retired adults (Clusters 2 & 4)**, who have high purchasing power and low risk.
4. **Allocate marketing budgets conservatively toward Cluster 5** — highest loss rate and lowest profitability make heavy investment inefficient.
5. **Cross-sell premium products to Cluster 4** — highest spending power and lowest loss percentage make them ideal upsell candidates.
6. **Target middle-aged individuals for motor insurance**, and young adults earning ~€2,000/mo for household and work compensation insurance.

---

## Key Analytical Findings

- Customer **commitment** (premium spend relative to income) increases with age.
- **Household premiums** are the dominant product category across most segments and represent the largest growth lever.
- Customers with **basic education** allocate the highest share of their budget to household premiums.
- **Motor premiums** are highest among middle-aged working professionals (Cluster 3).
- **Geographic area** does not meaningfully influence the percentage of premiums paid.
- **Teenagers** show the highest median household premiums of any age class.
---

```

---

## Repository Structure

```
insurance-customer-segmentation-ml/
│
├── code.ipynb                   # Main notebook (full pipeline)
├── data/insurance_dataset.csv   # Input dataset 
├── images                       # charts and Visualisation  
├── requirements.txt             # Python dependencies
└── README.md
```

---


