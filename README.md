# RevTrend-AI-Customer-Segmentation-Revenue-Risk-System
An interactive dashboard for identifying revenue decline, customer segmentation, and generating insights with LLMs, enabling businesses to take timely, data-driven action.

## 🌐 Live Demo : https://revenue-risk-system.streamlit.app

## What Business Problem Does It Solve?
### Businesses often struggle to:

- **Identify truly declining customers** in the presence of noise, volatility, or seasonality
- **Prioritise at-risk customers** from large portfolios
- Understand **behavioural patterns across customer segments**
- **Communicate insights across teams**

### This system addresses these challenges by combining:
1. **Revenue Trend Detection** – Uses **STL decomposition** to isolate long-term revenue trends from seasonal or irregular variations
2. **Customer Segmentation** – Applies **KMeans clustering** on trend-based metrics (e.g. slope, revenue volatility, activity span) to group customers by behavioural patterns
3. **Anomaly Detection** – Identifies abnormal months using **residual analysis** (z-score ), helping flag potential red alerts for follow-up
4. **Actionable Insight Generation** – Uses **LLMs (OpenAI)** to convert statistical patterns into natural language summaries with recommended business actions

## Key Features
### 1. Revenue Risk Classification

- Automatically assigns customers into four decline levels (0 = stable, 3 = sharply declining) based on slope and volatility.

 ### 2. Customer Segmentation Insights

- Uses K-Means clustering to group customers based on revenue trends, volatility, and activity.
- Each cluster is visualised with a radar chart (Z-score normalised) to compare behavioural profiles.
- LLM-generated summaries explain each segment’s characteristics and recommend business actions

  **Figure 1. Cluster Comparison Radar Charts (Z-score Normalised)**  
  This visual compares customer behavioural patterns across clusters based on revenue slope, volatility, and activity level. Cluster 2 shows distinctive traits, while Clusters 0 & 1 are more similar.
  <img width="849" height="438" alt="image" src="https://github.com/user-attachments/assets/dcac1c5c-debd-46b4-8ecf-b47b51d06bc3" />


### 3. Interactive Trend Diagnostics

- View individual customer revenue lines with trend lines, moving averages, and anomaly markers.

  **Figure 2. Interactive Trend Diagnostics Module**

  This module enables users to inspect individual customers’ revenue history with automatic anomaly detection.
  Trend lines (Theil-Sen), 3-month moving averages, and red anomaly flags provide interpretability for revenue decline patterns.


  <img width="788" height="649" alt="image" src="https://github.com/user-attachments/assets/5f45ea02-4cd6-4aed-8541-c9cb7f0cf62b" />


### 4. High-Risk Customer Table

- Enables users to filter and explore high-risk customers by decline level (e.g., Level 2 = moderate decline, Level 3 = critical decline).
- Displays a summary table of key metrics for each customer, including slope, average revenue, and activity duration.
- Allows users to drill down into individual customer profiles to view revenue trends and LLM-generated business summaries.

  **Figure 3. High-Risk Customer Table and Drill-Down Interface**  

  
  <img width="789" height="555" alt="image" src="https://github.com/user-attachments/assets/2bc52cca-ba25-42d7-aac0-d27c52362868" />


  <img width="811" height="628" alt="image" src="https://github.com/user-attachments/assets/08ed4038-10c0-486d-8d9d-3c67e4dc6457" />



### 5. Auto-Generated Group Reports

- Allows users to select a customer segment and decline level to generate real-time LLM insights based on statistical metrics such as revenue, slope, and volatility.
- Results are summarised into business-friendly markdown reports and can be downloaded in Markdown or TXT format.

   **Figure 4. Group-Level Metrics Used for LLM Summarisation**  

<img width="741" height="450" alt="image" src="https://github.com/user-attachments/assets/e37ad9e4-471d-4064-ba94-fde78c2eba8e" />





  
## 🧠 Key Techniques
| Technique              | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| STL decomposition       | Isolate true revenue trend from seasonal patterns |
| Theil–Sen regression   | Robust slope estimation                           |
| Z-score              | Identify outliers and customer volatility         |
| K–Means Clustering   | Segment customers by behaviour                    |
| OpenAI GPT API       | Auto-generate business summaries                  |
