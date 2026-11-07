import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Define the exact path to your CSV file
# Replace the path below with the actual location of your CSV file
csv_path = r"C:\Users\Asmaa\PycharmProjects\credit_risk_project_xai\outputs\results\shap_lime_comparison.csv"

# Optional: Check if the file exists before trying to read it
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    print("Please check the file path and try again.")
else:
    # 2. Load the comparison data
    df = pd.read_csv(csv_path)

    # 3. Prepare the data for plotting
    overlap_data = []

    for risk_cat in ['LOW_RISK', 'HIGH_RISK', 'BORDERLINE']:
        subset = df[df['risk_category'] == risk_cat]

        shap_only = len(subset[(subset['in_shap_top5'] == True) & (subset['in_lime_top5'] == False)])
        lime_only = len(subset[(subset['in_shap_top5'] == False) & (subset['in_lime_top5'] == True)])
        both = len(subset[(subset['in_shap_top5'] == True) & (subset['in_lime_top5'] == True)])

        # Format the risk category name for better display
        if risk_cat == 'LOW_RISK':
            name = 'Low Risk\n(SK_ID: 371134)'
        elif risk_cat == 'HIGH_RISK':
            name = 'High Risk\n(SK_ID: 376872)'
        else:
            name = 'Borderline Risk\n(SK_ID: 252790)'

        overlap_data.append({
            'Applicant': name,
            'SHAP Only': shap_only,
            'LIME Only': lime_only,
            'Overlap (Both)': both
        })

    plot_df = pd.DataFrame(overlap_data)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    melted_df = plot_df.melt(id_vars='Applicant', var_name='Category', value_name='Count')

    ax = sns.barplot(
        data=melted_df,
        x='Applicant',
        y='Count',
        hue='Category',
        palette={'SHAP Only': '#4C72B0', 'LIME Only': '#C44E52', 'Overlap (Both)': '#55A868'},
        edgecolor='black'
    )

    # 5. Customize the chart
    plt.title('Figure 4.15: SHAP and LIME Top-5 Feature Overlap by Applicant Risk Category',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Applicant', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.ylim(0, 6)

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=11, fontweight='bold', padding=3)

    plt.legend(title='Feature Source', title_fontsize=11, fontsize=10, loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save to your project folder
    save_path = r"C:\Users\Asmaa\PycharmProjects\credit_risk_project_xai\Figure_4.15_SHAP_LIME_Overlap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure generated and saved to: {save_path}")

    plt.show()