# Perovskite Solar Cell Stability & Efficiency Predictor

A **Streamlit-based web application** to predict the **stability** and **efficiency** of perovskite solar cells using trained machine learning models.

---

##  Overview

Perovskite solar cells are among the most promising photovoltaic technologies, offering high efficiency and low production costs. However, challenges in long-term **stability** and **performance optimization** remain.

This tool allows researchers and engineers to:

-  Predict **stability scores** based on material compositions and environmental conditions.
-  Estimate **efficiency (%)** based on fabrication parameters.
- ⏱ Quickly screen experimental designs using ML.

---

## ⚙️ Features

Predicts **Stability** using:
- A-site, B-site, and C-site ion selection  
- Synthesis relative humidity (%)  
- Bandgap (eV)  
-  Includes interaction feature: `Bandgap × Humidity`

Predicts **Efficiency** using:
- Perovskite thickness  
- Annealing time  
- Deposition temperature  
- Bandgap, HTL & ETL materials, and their thickness  
- Includes interaction features: `Thickness × Annealing`, `Bandgap × Annealing`

 Built with:
- **Python**
- **Streamlit** (web frontend)
- **Scikit-learn** (ML models)
- **Pickle** (model serialization)

---
## Performance
| Model       | R² Score | RMSE   |
|-------------|----------|--------|
| Stability   | 0.59     | 2.24   |
| Efficiency  | 0.60     | 0.90   |
⸻

## License

This project is for educational and research purposes only.
All rights reserved © 2025.

⸻

## Developed By

Tiya
(B.Tech, NIT Mizoram)
Machine Learning and  Materials Science Research Intern @IIT Guwahati


⸻

## Live Demo 

Click here to try the app live
https://perovskite-solar-cell-predictor-wgxqfqjxkislntydctmruj.streamlit.app/

