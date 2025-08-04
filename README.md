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
**Revenue Risk Classification**

- Automatically assigns customers into four decline levels (0 = stable, 3 = sharply declining) based on slope and volatility.

 **Customer Segmentation Insights**

- Uses K-Means clustering to group customers based on revenue trends, volatility, and activity.
- Each cluster is visualised with a radar chart (Z-score normalised) to compare behavioural profiles.
- LLM-generated summaries explain each segment’s characteristics and recommend business actions

**Interactive Trend Diagnostics**

- View individual customer revenue lines with trend lines, moving averages, and anomaly markers.

**High-Risk Customer Table**

- Filter and explore customers by decline level, and select a specific customer to view trend charts and LLM-generated business summaries.

**Auto-Generated Group Reports**

- Allows users to select a customer segment and decline level to generate real-time LLM insights based on statistical metrics. Results can be downloaded in Markdown or TXT format.
  
## 🧠 Key Techniques
| Technique              | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| STL decomposition       | Isolate true revenue trend from seasonal patterns |
| Theil–Sen regression   | Robust slope estimation                           |
| Z-score              | Identify outliers and customer volatility         |
| K–Means Clustering   | Segment customers by behaviour                    |
| OpenAI GPT API       | Auto-generate business summaries                  |
