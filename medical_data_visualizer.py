import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
def load_and_preprocess_data():
    print("📂 Loading data...")
    df = pd.read_csv("medical_examination.csv")
    print("✅ Data loaded successfully!")

    # Calculate BMI and determine if overweight
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['overweight'] = (df['BMI'] > 25).astype(int)

    # Normalize cholesterol and glucose (0 = good, 1 = bad)
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    print("✅ Data preprocessing complete!")
    return df

# Draw Categorical Plot
def draw_cat_plot(df):
    print("📊 Generating categorical plot...")
    
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Create categorical plot
    fig = sns.catplot(x='variable', y='total', hue='value', kind='bar', col='cardio', data=df_cat)
    
    print("✅ Categorical plot created!")
    return fig

# Draw Heat Map
def draw_heat_map(df):
    print("🔥 Generating heatmap...")
    
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", vmax=0.3, center=0, square=True, linewidths=0.5)

    print("✅ Heatmap created!")
    return fig

# Main execution
if __name__ == "__main__":
    print("🚀 Script Started!")
    df = load_and_preprocess_data()
    draw_cat_plot(df)
    draw_heat_map(df)
    print("🏁 Script finished!")
