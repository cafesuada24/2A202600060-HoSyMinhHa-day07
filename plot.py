import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the results
csv_filename = "retrieval_metrics_summary.csv"
df = pd.read_csv(csv_filename)

# 2. Reshape the data for plotting (melt)
# This converts the wide format (columns of metrics) into a long format
df_melted = df.melt(id_vars="Strategy", var_name="Metric", value_name="Score")

# 3. Set up the plotting canvas and style
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid", palette="muted")

# 4. Create the grouped bar chart
ax = sns.barplot(
    data=df_melted, 
    x="Metric", 
    y="Score", 
    hue="Strategy",
    edgecolor="black",
    linewidth=0.5
)

# 5. Formatting and Aesthetics
plt.title("Retrieval Evaluation Metrics by Splitting Strategy", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Evaluation Metric", fontsize=12, fontweight='bold')
plt.ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight='bold')
plt.ylim(0, 1.05) # Lock y-axis to the 0-1 range

# Move the legend outside the plot
plt.legend(title="Splitting Strategy", title_fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left')

# 6. Add exact score labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    # Only label bars that actually have a height to avoid clutter
    if height > 0.01: 
        ax.annotate(f"{height:.2f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    fontsize=9, color='black', 
                    xytext=(0, 4), textcoords='offset points')

# Adjust layout to prevent cutting off the legend
plt.tight_layout()

# Save and show
output_filename = "retrieval_metrics_chart.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved successfully as '{output_filename}'")

plt.show()
