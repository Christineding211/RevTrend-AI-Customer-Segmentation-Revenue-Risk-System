
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import os
from openai import OpenAI
from dotenv import load_dotenv


st.set_page_config(page_title="📊 RevTrend: AI Customer Segmentation & Revenue Risk Dashboard",layout="centered")

def load_data():
    try:
        result_df = pd.read_csv("result_df.csv")
        df_pivot  = pd.read_csv("df_pivot.csv", index_col=0)
        df_final = pd.read_csv("df_final.csv", index_col=0)
        with open("customer_llm_summaries_clean.json", "r", encoding="utf-8") as f:
            cust_llm = json.load(f)
        with open("llm_cluster_responses.json", "r", encoding="utf-8") as f:
            cluster_llm = json.load(f)
        return result_df, df_pivot, df_final,cust_llm, cluster_llm
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop() 

# 正確接收回傳值
result_df, df_pivot, df_final,cust_llm, cluster_llm = load_data()


st.title("📊 Advanced Customer Revenue Trend & Segmentation Dashboard")

#Picture 1 
st.subheader("1️⃣ Revenue Risk Overview")
level_counts = result_df["decline_level"].value_counts().sort_index()
cols = ["#8CB5F6", "#E4BEF5", "#F6CD82", "#FA6986"]        # green → red

c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(4, 3))

    x = level_counts.index
    y = level_counts.values

    ax.bar(x, y, color=cols, width=0.6, align='center')
    ax.set_xticks([0, 1, 2, 3])  
    ax.set_xticklabels(['0', '1', '2', '3'])  
    ax.set_xlabel("Decline Level (0 = Stable, 3 = Critical)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Decline Level Distribution", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width= False) 



with c2:
    st.metric("🔢 Total Customers", len(result_df))
    st.metric("🔴 High-Risk (Lvl 2+3)", level_counts.get(2, 0) + level_counts.get(3, 0))
    st.metric("📈 Avg Slope (All)", f"{result_df['slope'].mean():.1f}")

st.divider()

#picture2

st.subheader("2️⃣ Cluster Insights")

# 
features = ["revenue_mean", "slope", "cv", "resid_std", "active_months"]

# cluster mean
summary = result_df.groupby("cluster")[features].mean()

#  log1p
summary_log = summary.copy()
summary_log[["revenue_mean", "resid_std"]] = np.log1p(summary_log[["revenue_mean", "resid_std"]])
#  Z-score 
scaler = StandardScaler().fit(summary_log)
summary_z = pd.DataFrame(
    scaler.transform(summary_log),
    index=summary_log.index,
    columns=summary_log.columns
)

# Radar chart 
def build_radar(df, title, r_range=[-2, 2], showlegend=True):
    fig = go.Figure()
    cluster_colors = {
        0: "#75F4C5",  # light green
        1: "#7BD2F1",  # light blue
        2: "#EDEF5F",  # light yellow
    }
    for cid, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r = row.tolist() + [row[0]], 
            theta = features + [features[0]],
            name = f"Cluster {cid}",
            fill = 'toself',
            opacity = 0.4,
            line = dict(color=cluster_colors.get(cid, "#CCCCCC"))  
        ))
    fig.update_layout(
        title = title,
        polar = dict(radialaxis = dict(range=r_range, visible=True)),
        showlegend = showlegend
    )
    return fig

# Step 6: Plot radar and LLM summary side-by-side
# Cluster 0 & 1
sub_left = summary_z.loc[[0, 1]]
fig_left = build_radar(sub_left, "Cluster 0 & 1 (Z-score)", [-2, 2])


# Cluster 2
sub_right = summary_z.loc[[2]]
fig_right = build_radar(sub_right, "Cluster 2 (Z-score)", [-2, 2], showlegend=False)


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_left, use_container_width=True)
with col2:
    st.plotly_chart(fig_right, use_container_width=True)


# 🔹 Load cluster-level LLM summary
st.markdown("---")
st.markdown("## 🧠 LLM Cluster Summaries")

with open("llm_cluster_responses.json", "r", encoding="utf-8") as f:
    cluster_summary = json.load(f)

for cid in [0, 1, 2]:
    st.markdown(f"### 🔹 Cluster {cid}")
    st.markdown(cluster_summary.get(str(cid), "_No summary available_"))


#section 3


st.markdown("<span style='font-size:24px'><b>3. Individual Customer Revenue Trend & Anomaly Highlights</b></span>", unsafe_allow_html=True)

#  Decline level 
sel_level = st.selectbox("Select Decline Level", sorted(df_final['decline_level'].dropna().unique()),key="sel_level_sectionA")
sel_customers = df_final[df_final["decline_level"] == sel_level].index.tolist()
sel_cust = st.selectbox("Select Customer ID", sel_customers)

#  
month_cols = [col for col in df_final.columns if col.startswith("20")]
rev = df_final.loc[sel_cust, month_cols].astype(float)
rev = rev[rev.notna()]

#  
x = np.arange(len(rev))
y = rev.values
slope, intercept, *_ = linregress(x, y)
trend = intercept + slope * x
ma3 = rev.rolling(3).mean()

# Detect anomalies and negatives
resid = y - trend
resid_std = resid.std()
anomaly_mask = np.abs(resid) > 1.5 * resid_std
negative_mask = y < 0


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rev.index, y, label="Monthly Revenue", marker='o')
ax.plot(rev.index, trend, label="Trend Line (Theil-Sen)", linestyle='--')
ax.plot(rev.index, ma3, label="3-Month Moving Average", linestyle=':')

# resid
ax.scatter(rev.index[anomaly_mask], y[anomaly_mask], color='red', label='Anomaly (high resid)', zorder=5)

# 🟥 revenue be negative
ax.scatter(rev.index[negative_mask], y[negative_mask], facecolors='none',
           edgecolors='red', s=100, linewidths=1.5, label='Negative Revenue')

ax.set_title(f"Revenue Trend for Customer {sel_cust}",fontsize=14)
ax.set_ylabel("Revenue (£)",fontsize=12)
ax.set_xlabel("Month",fontsize=12)
plt.xticks(rotation=45, fontsize=10)
ax.legend()
ax.grid(True)
plt.tight_layout()
#set in the middle
col1, col2, col3 = st.columns([1,2,1])
st.pyplot(fig, use_container_width= False)

#Section 4
st.markdown("## 🔴 Section B: High-Risk Customer Overview (Decline Level 2 or 3)")

# Step 1️⃣ Decline level selector
sel_level = st.selectbox("Select Decline Level", [2, 3],key="sel_level_sectionB")

# Step 2️⃣ Filter for high-risk customers
df_risk = df_final[df_final['decline_level'] == sel_level].copy()

# Display basic customer metrics
st.markdown(f"Identified {len(df_risk)} high-risk customers with Decline Level {sel_level}:")
st.dataframe(df_risk[["cluster", "slope", "revenue_mean", "active_months"]])

# Step 3️⃣ Expanded customer-level insights
st.markdown("---")
st.markdown("### 🔍 Customer Trend & LLM Summary")

# Select individual customer to inspect
sel_cust = st.selectbox("Select a customer to inspect", df_risk.index.tolist())

# Extract revenue time series
rev = df_final.loc[sel_cust, month_cols].astype(float)
rev = rev[rev.notna()]

# Trend calculation
x = np.arange(len(rev))
y = rev.values
slope, intercept, *_ = linregress(x, y)
trend = intercept + slope * x
ma3 = rev.rolling(3).mean()

# Revenue trend chart
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(rev.index, y, label='Monthly Revenue (£)', marker='o')
ax.plot(rev.index, trend, label='Trend Line (Theil-Sen)', linestyle='--')
ax.plot(rev.index, ma3, label='3-Month Moving Average', linestyle=':')
ax.set_title(f"Customer {sel_cust} – Revenue Trend")
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig, use_container_width=True)

# LLM summary display
st.markdown("### 🧠 LLM Summary")
summary = cust_llm.get(str(sel_cust), "_No summary available._")
st.markdown(summary)

# === Load/OpenAI setup ===

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SUMMARY_CACHE_PATH = "group_llm_summaries.json"
if os.path.exists(SUMMARY_CACHE_PATH):
    with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
        summary_cache = json.load(f)
else:
    summary_cache = {}

# === SECTION C ===
st.markdown("## 📄 Section C: Auto Report Generator")

# Step 1: User selection
sel_cluster = st.selectbox("Select Cluster", sorted(result_df["cluster"].dropna().unique()),key="cluster_selector_sectionC")
sel_level = st.selectbox("Select Decline Level", sorted(result_df["decline_level"].dropna().unique()),key="level_selector_sectionC")

# Step 2: Filter group
df_group = result_df[(result_df["cluster"] == sel_cluster) & (result_df["decline_level"] == sel_level)]

if df_group.empty:
    st.warning("No customers found for this group.")
    st.stop()

# Step 3: Summary statistics
summary_stats = df_group[["revenue_mean", "slope", "cv", "resid_std", "active_months"]].mean()
n_customers = len(df_group)

st.markdown("### 📊 Group Statistics")
st.dataframe(summary_stats.to_frame("Mean Value"))

# Step 4: Generate unique group key
group_key = f"cluster{sel_cluster}_level{sel_level}"

# Step 5: Check cache
if group_key not in summary_cache:
    st.warning("⚠️ No cached summary available for this group.")
    if st.button("🔁 Generate Summary with LLM"):
        with st.spinner("Generating summary..."):
            prompt = f"""
You are a business analyst.

Based on the following group metrics of {n_customers} customers, write a short business summary with the following sections:
1. Segment Profile
2. Risk Analysis
3. Suggested Action

Group Statistics:
- Avg Revenue: £{summary_stats['revenue_mean']:.2f}
- Trend Slope: {summary_stats['slope']:.2f}
- Coefficient of Variation (CV): {summary_stats['cv']:.2f}
- Residual Std: {summary_stats['resid_std']:.2f}
- Active Months: {summary_stats['active_months']:.0f}
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional business analyst."},
                    {"role": "user", "content": prompt}
                ]
            )

            llm_summary = response.choices[0].message.content.strip()
            summary_cache[group_key] = llm_summary

            # Save to cache file
            with open(SUMMARY_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(summary_cache, f, indent=2, ensure_ascii=False)
            
            st.success("✅ Summary successfully generated and cached!")
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except AttributeError:
                    st.warning("⚠️ Cannot rerun automatically. Please manually refresh the page.")


# Step 6: Display report
if group_key in summary_cache:
    report_md = f"""### 📊 Cluster {sel_cluster} | Decline Level {sel_level} – Summary Report

**Group Size:** {n_customers} customers

**Mean Metrics:**
- Avg Revenue (£): {summary_stats['revenue_mean']:.2f}
- Trend Slope: {summary_stats['slope']:.2f}
- CV: {summary_stats['cv']:.2f}
- Residual Std Dev: {summary_stats['resid_std']:.2f}
- Active Months: {summary_stats['active_months']:.0f}

**🧠 LLM Summary:**
{summary_cache[group_key]}
"""

    st.markdown("### 📝 Generated Report")
    st.code(report_md, language="markdown")

    st.download_button("📥 Download Report as .txt", data=report_md, file_name=f"{group_key}_summary.txt")





