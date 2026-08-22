# logistics-data-analyst-internship-week1
# Logistics Data Analyst Internship

## Week 1 – Strategic Planning and Data Exploration in Logistics

### Project Overview

This repository contains my work for the Yuva Intern Logistics Data Analyst Internship.

The project focuses on an e-commerce logistics scenario involving delivery performance, transportation cost, operational efficiency, and data-driven decision making.

For academic purposes, the case company used in this project is **SwiftCart Logistics**, a fictional mid-sized e-commerce logistics company. The company name and business situation are assumed only for the internship case study. Public data will be used for the actual analysis.

---

## Business Problem

The project investigates whether historical order and delivery data can help explain differences in delivery performance and logistics cost.

The main questions are:

* Which factors are associated with late deliveries?
* How does delivery performance vary across locations, sellers, products, and order conditions?
* How does freight cost change with shipment characteristics such as value, weight, or destination?
* Which fields can be used later for prediction or optimisation?

---

## Week 1 Objectives

1. Measure delivery performance.
2. Identify patterns behind delivery delays.
3. Study logistics and freight cost.
4. Prepare data for future predictive analysis.
5. Present findings in a form that can be understood by non-technical stakeholders.

---

## KPIs Planned

| KPI                    | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| On-time Delivery Rate  | Measures how often the delivery promise is met |
| Average Delivery Time  | Measures typical delivery duration             |
| Average Delivery Delay | Measures the size of late-delivery problems    |
| Freight Cost per Order | Measures transportation cost pressure          |
| Late-order Share       | Measures the proportion of late deliveries     |

Final KPI definitions will be confirmed after the Week 2 data-quality and dataset-structure checks.

---

## Data Source

The planned implementation dataset is the **Brazilian E-Commerce Public Dataset by Olist**.

The dataset contains information related to:

* Orders
* Customers
* Sellers
* Products
* Payments
* Reviews
* Order items
* Delivery estimates and dates
* Freight values

The dataset will be inspected, cleaned, and transformed during Week 2 before the main analysis is performed.

---

## Data Science Approach

The project follows a staged analytics approach:

**Descriptive Analysis → Exploratory Data Analysis → Prediction → Clustering → Optimisation**

The exact modelling techniques will be selected after the dataset is cleaned and explored.

Python and the `scikit-learn` ecosystem are planned for the later analytical and modelling stages.

---

## Four-Week Roadmap

### Week 1 – Plan

Define the business scenario, KPIs, data requirements, analytical methods, and project roadmap.

**Deliverable:** Week 1 Project Report

### Week 2 – Prepare

Collect and inspect the public dataset, understand table relationships, clean fields, handle missing values, and create analysis-ready features.

**Expected output:** Clean dataset and data dictionary

### Week 3 – Analyse

Perform exploratory data analysis, compare KPIs, create visualisations, and identify important operational patterns.

**Expected output:** Notebook, charts, or dashboard

### Week 4 – Model and Recommend

Develop a suitable prediction, segmentation, or optimisation component and convert the findings into practical recommendations.

**Expected output:** Final analysis and recommendations

---

## Week 1 Deliverable

The current Week 1 submission is available here:

**[Week 1 Project Report](./Week_1.docx)**

The report covers:

* Background and business scenario
* Project objectives
* KPI planning
* Public dataset research
* Data science methodology
* Four-week roadmap
* Initial Python approach
* Data-quality considerations
* Expected outcomes
* References

---

## Important Note

Week 1 is a **planning stage**. No final numerical business findings are presented at this stage.

The actual dataset analysis will begin in Week 2 after the public data is collected, inspected, cleaned, and transformed.

---

## Repository Structure

```text
logistics-data-analyst-internship/
│
├── README.md
│
├── Week_1/
│   └── Week_1.docx
│
├── Week_2/
│   └── (to be added)
│
├── Week_3/
│   └── (to be added)
│
└── Week_4/
    └── (to be added)
```

---

## References

1. Olist – Brazilian E-Commerce Public Dataset
2. scikit-learn documentation
3. AWS Supply Chain Calculations
4. NIST Advanced Manufacturing Series 100-75

---

## Author

**Ayushman Singh**
BCA Student | Aspiring Data Analyst
Yuva Intern – Logistics Data Analyst Internship
