
# 🏏 Durham Junior Cricket Analysis

**Durham Junior Cricket Analysis** is a pipeline built to validate Under‑15 selection methods, created as part of a **Master of Data Science** dissertation at **Durham University (2025)**.

---

## 📝 Overview

This project leverages **Python**, **R**, and **Jupyter** tools to evaluate and improve the player selection process for junior cricket in Durham.  
It focuses on:

- Cleaning and preprocessing batting, bowling, and fielding datasets  
- Performing **PCA** (Principal Component Analysis) and **clustering** to identify player groups  
- Building a **Streamlit dashboard** for visual exploration  
- Documenting findings in the full dissertation

---

## 📂 Repository Structure

| File / Folder                           | Purpose                                                   |
| -------------------------------------- | -------------------------------------------------------- |
| `dissCode.ipynb`                       | Jupyter Notebook containing the main analysis workflow   |
| `clustering_analysis.R`                | R script for clustering and exploratory analysis         |
| `pca_analysis.R`                       | R script for PCA and dimensionality reduction            |
| `dashboard.py`                         | Python dashboard for performance visualizations          |
| `batting.xlsx`, `bowling.xlsx`, `fielding.xlsx` | Raw performance datasets                     |
| `cdata.csv`, `final_data*.csv`, `top20_players.csv` | Intermediate & final processed datasets |
| `requirements.txt`                     | Python dependencies                                      |
| `Dissertation_Z0207662.pdf`            | Full dissertation write‑up (2025)                        |

---

## 🚀 Getting Started

### **Prerequisites**

Make sure you have the following installed:

- **Python** ≥ 3.8
- **R** (latest stable release)
- **pip** for installing Python packages

### **Installation**

Clone the repository:

```bash
git clone https://github.com/kodiakthebear/durham-junior-cricket-analysis.git
cd durham-junior-cricket-analysis
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For R dependencies, install via CRAN:

```R
install.packages(c("tidyverse", "cluster", "factoextra"))
```

---

## 🧩 How to Run the Analysis

### **1. Explore the Jupyter Notebook**
```bash
jupyter notebook dissCode.ipynb
```

### **2. Run Clustering Analysis**
```bash
Rscript clustering_analysis.R
```

### **3. Perform PCA (Dimensionality Reduction)**
```bash
Rscript pca_analysis.R
```

### **4. Launch the Dashboard**
```bash
python dashboard.py
```

### **5. Review the Full Findings**
Refer to the dissertation for context:  
[`Dissertation_Z0207662.pdf`](./Dissertation_Z0207662.pdf)

---

## 📊 Results & Outputs

- **Player Clusters** → Identifies groups based on performance
- **Dimensionality Reduction** → Simplifies player attributes using PCA
- **Top 20 Player Selections** → Compares model-predicted selections vs. actual picks
- **Visual Insights** → Interactive dashboards for exploring stats

---

## 🖥️ Running the Streamlit Dashboard (macOS)

The repository includes an interactive **Streamlit dashboard** (`dashboard.py`) that visualises player performance metrics, clustering insights, and top player selections.

### **1. Navigate to the Project Directory**
```bash
cd durham-junior-cricket-analysis
```

### **2. Create Virtual Environment (not required)***
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run Dashboard (Requires Streamlit + All libraries and dependencies)**
```bash
streamlit run dashboard.py
```

### **5. Upload Data File**
```
cdata.csv (in repo) or any file with the same format (same columns)
```
---
## 🎯 Project Goals

This analysis aims to:

- Provide **data-driven** insights for junior cricket player selection  
- Evaluate the effectiveness of **machine learning** in sports analytics  
- Offer a **reproducible pipeline** for future Durham Cricket evaluations

---

## 🧑‍💻 Tech Stack

```yaml
Languages:
  - Python 3.8+
  - R (CRAN)

Key Libraries:
  Python:
    - pandas
    - scikit-learn
    - xgboost
    - matplotlib
    - seaborn
    - streamlit
  R:
    - tidyverse
    - cluster
    - factoextra
```

---

## 🤝 Credits

Developed by **Mukund Tiwari** *(kodiakthebear)*  
Part of the **Master of Data Science** dissertation at **Durham University**.

For discussions, collaborations, or cricket analytics nerdery:  
[GitHub Profile](https://github.com/kodiakthebear)

---

> “Data beats instinct — but together, they’re unstoppable.”

