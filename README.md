# 🛍️ Customer Segmentation Analysis: Online Retail (RFM + K-Means)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E?logo=scikitlearn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-notebook-F9AB00?logo=googlecolab&logoColor=white)

> Segmenting **4,373 customer profiles** from over **540,000 e-commerce transactions** into six behavioural groups using RFM and behavioural features, then translating the segments into retention, upsell and targeting recommendations.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Questions](#-business-questions)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Recommended Actions by Segment](#-recommended-actions-by-segment)
- [Limitations and Next Steps](#-limitations-and-next-steps)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 📌 Project Overview

Not every customer is worth the same marketing effort. This project takes raw, messy transaction logs from an online retailer and answers a practical question: **who are our customers, how do they behave, and how should we treat each group differently?**

The workflow covers the full analytics cycle:

1. Data investigation and cleaning (missing IDs, cancellations, discounts, postage lines)
2. Feature engineering at transaction and customer level
3. RFM and behavioural feature construction
4. K-Means clustering with elbow and silhouette analysis
5. Cluster profiling, product preference and cross-sell analysis
6. Loyalty, churn and repeat-purchase metrics

---

## ❓ Business Questions

| # | Question |
|---|----------|
| 1 | Who are our most valuable customers, and what do their purchase patterns look like? |
| 2 | What distinct customer segments exist, and do they prefer different products? |
| 3 | How can each segment be targeted effectively? |
| 4 | Where is there room to grow revenue per customer (upsell / cross-sell)? |
| 5 | What characterises loyal customers, and how do we reduce churn? |

---

## 📂 Dataset

- **Source:** [Online Retail dataset on Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data/data) (originally from the UCI Machine Learning Repository)
- **Content:** Real transactions from an online retailer between December 2010 and December 2011
- **Size:** 541,909 rows × 8 columns

| Column | Description |
|--------|-------------|
| `InvoiceNo` | Invoice identifier (a `C` prefix marks a cancellation) |
| `StockCode` | Product code (also includes non-product codes such as `POST` and `D`) |
| `Description` | Product name |
| `Quantity` | Units per line (negative for returns/cancellations) |
| `InvoiceDate` | Date and time of the transaction |
| `UnitPrice` | Price per unit |
| `CustomerID` | Customer identifier |
| `Country` | Customer country |

> The raw data is not included in this repo. Download it from Kaggle (see [How to Run](#-how-to-run)).

---

## 🔬 Methodology

### 1. Data investigation and cleaning

- **Missing `CustomerID` (135,080 rows, ~24.9%)**: investigated against zero-price rows. Only 2,475 rows had both a missing ID and a zero price, so most missing IDs belong to genuine guest/anonymous purchases. These rows were retained and labelled `Missing_CustID` rather than dropped, to avoid discarding a quarter of the data.
- **Missing `Description` (1,454 rows)**: imputed using a `StockCode` → description lookup; anything still missing was labelled `Unknown Product`.
- **Cancellations**: invoices starting with `C` (9,288 line items across 3,836 invoices) were confirmed to always carry negative quantities, so they were flagged as cancellations.
- **Non-product stock codes**: `D` (77 "Discount" lines) and `POST` (postage) were identified as administrative charges, not products.

### 2. Feature engineering (transaction level)

`Amount`, `Revenue`, `is_return`, `is_cancellation`, `transaction_type` (Sale / Return / Cancellation / Discount / Postage), `is_special_stock`, invoice-level value, and date parts (year, month, day, weekday, hour).

### 3. Customer-level features

Aggregated per customer, with recency measured from the day after the last invoice in the data:

| Feature | Meaning |
|---------|---------|
| `recency` | Days since last purchase |
| `frequency` | Number of distinct invoices |
| `monetary` | Net spend (returns included) |
| `avg_basket_value` | Average invoice value |
| `avg_item_qty` | Average units per line |
| `num_distinct_products` | Breadth of products bought |
| `return_rate` | Returned lines ÷ total lines |
| `discount_share`, `postage_share` | Net discount and postage amounts |
| `avg_unitprice` | Average price point of items bought |

### 4. Preprocessing and clustering

- Sign-preserving **log transform** (`log1p`) on skewed features to reduce the pull of extreme values
- **StandardScaler** so no feature dominates by scale
- **K-Means** tested for k = 2 to 10, using the **elbow method** and **silhouette score**
- Final model: **k = 6** (`random_state=42`, `n_init=20`)

<!-- Add your elbow/silhouette chart here once exported, e.g.:
![Elbow and silhouette](images/elbow_silhouette.png)
-->

---

## 📊 Key Findings

### Customer segments

| Cluster | Suggested label | Customers | Avg recency (days) | Avg frequency (orders) | Avg net spend | Return rate |
|:------:|-----------------|----------:|-------------------:|-----------------------:|--------------:|------------:|
| 5 | **VIP Champions** | 22 | 15.2 | 46.4 | 44,875.76 | 5.9% |
| 2 | **Continental European Regulars** | 306 | 69.6 | 6.3 | 2,748.07 | 2.5% |
| 1 | **Active Core Customers** | 2,399 | 39.9 | 6.8 | 2,453.37 | 2.0% |
| 3 | **Low-Value, Lapsing Customers** | 1,593 | 171.5 | 1.8 | 376.23 | 2.5% |
| 0 | **Return-Heavy Customers** | 52 | 227.9 | 1.6 | −251.73 | 83.2% |
| 4 | *Guest-checkout placeholder (artifact)* | 1 | 1.0 | 3,710 | 1,447,682.12 | 1.3% |

> Segment labels are my interpretation of each cluster's profile; the clustering itself is unlabelled.

**Segment highlights**

- **VIP Champions (22 customers):** tiny group, huge impact. Very recent, extremely frequent and by far the highest spenders. Mostly UK-based.
- **Active Core (2,399):** the backbone of the business. Recent (~40 days), repeat buyers with a very low return rate, almost entirely UK.
- **Continental European Regulars (306):** led by Germany, France and Belgium. Spend similar to the core group but less recent; postage charges are the single largest revenue line for this segment, so shipping matters to them.
- **Low-Value, Lapsing (1,593):** mostly one- or two-time buyers who haven't returned in ~170 days.
- **Return-Heavy (52):** about 1% of customers with an 83% return rate and negative net spend.
- **Cluster 4:** this is the `Missing_CustID` placeholder, not a real customer (see [Limitations](#-limitations-and-next-steps)).

### Other insights

- **Spend is highly skewed:** mean customer spend is ~2,229 against a median of ~648, so a minority of customers drives a large share of revenue.
- **Top customer:** excluding the placeholder, customer **14646 (Netherlands)** is the highest-value customer at ~279,489 net spend, followed by 18102 and 17450 (both UK).
- **Geography:** the UK generates ~8.19M in revenue, nearly 29× the next market (Netherlands, ~285K), followed by EIRE, Germany and France.
- **Repeat purchase rate:** **70%** of customers bought more than once.
- **Churn:** **19.8%** of customers had not purchased in the last 180 days.
- **Loyal vs. non-loyal** (loyal = 5+ orders and last purchase within 90 days):

  | | Avg spend | Avg recency (days) | Avg orders |
  |---|---:|---:|---:|
  | Loyal | 6,191 | 21.8 | 15.3 |
  | Not loyal | 641 | 120.1 | 2.2 |

  Loyal customers spend roughly **10× more**, and return rates are almost identical (3.3% vs 3.1%), so loyalty is driven by engagement, not by buying less carefully.
- **Cross-sell:** market-basket co-occurrence analysis surfaced frequently paired products, e.g. stock codes `22386` + `85099B` (833 invoices together) and `22697` + `22699` (784). These are strong candidates for bundles and "frequently bought together" prompts.

---

## 🎯 Recommended Actions by Segment

| Segment | Suggested approach |
|---------|--------------------|
| **VIP Champions** | Dedicated account management, early access to new products, loyalty perks; protect this group above all |
| **Active Core** | Cross-sell and bundle offers based on co-purchase pairs; nudge towards VIP behaviour with spend-threshold rewards |
| **Continental European Regulars** | Shipping incentives (free/discounted postage thresholds) and region-specific promotions |
| **Low-Value, Lapsing** | Win-back campaigns with time-limited offers; low-cost channels only, since value per customer is modest |
| **Return-Heavy** | Investigate causes (product quality, descriptions, sizing); consider return policy adjustments |

*These are analyst recommendations based on cluster profiles, not tested interventions.*

---

## ⚠️ Limitations and Next Steps

**Known limitations**

- **The `Missing_CustID` placeholder was clustered as if it were one customer**, producing the single-member Cluster 4 and slightly distorting the other clusters. It should be excluded before clustering.
- Extreme high-spend and negative-spend customers can still influence K-Means; capping or separate treatment may help.
- Churn (180 days) and loyalty (5+ orders, ≤90 days) thresholds are business assumptions, not validated against future behaviour.
- `monetary` is net of returns, so heavy returners can show negative values.

**Next steps**

- [ ] Remove `Missing_CustID` and re-run clustering; re-check the optimal k
- [ ] Visualise clusters in 2D with PCA (already imported in the notebook but not yet applied)
- [ ] Compare against alternatives (Gaussian Mixture, DBSCAN) and formal RFM scoring
- [ ] Estimate customer lifetime value with a probabilistic model (e.g. BG/NBD)
- [ ] Build an interactive dashboard (e.g. Power BI) for the segments

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn (StandardScaler, KMeans, silhouette_score)
- **Environment:** Google Colab / Jupyter Notebook

---

## 📁 Repository Structure

```
├── RetailerData.ipynb      # Full analysis notebook
├── data/                   # Place data.csv here (not included)
└── README.md
```

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/Lateephah/Online-retail-customer-segmentation.git
   cd Online-retail-customer-segmentation
   ```
2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. **Get the data:** download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data/data) and place it in `data/`.
4. **Update the data path:** the notebook was written in Google Colab and mounts Google Drive. If running locally, replace the Drive mount and read cells with:
   ```python
   df = pd.read_csv('data/data.csv', encoding='latin1')
   ```
5. **Run the notebook**
   ```bash
   jupyter notebook RetailerData.ipynb
   ```

---

## 👩🏾‍💻 Author

**Latifah Usaini Bashir**
Data Analyst | Computer Science

- GitHub: [@Lateephah](https://github.com/Lateephah)

If you found this project useful, feel free to ⭐ the repo.
