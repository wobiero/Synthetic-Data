import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Example Data
def create_real_data(n_patients=100):
    np.random.seed(42)
    
    # Generate patient data
    data = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(8, 2, n_patients).round(1),
        'weight_kg': np.random.normal(30, 5, n_patients).round(1),
        'baseline_6mwd': np.random.normal(350, 50, n_patients).round(1),
        'treatment_group': np.random.choice(['Treatment', 'Placebo'], n_patients),
        'exon_mutation': np.random.choice([45, 51, 53], n_patients, p=[0.4, 0.35, 0.25]),
        'completion_status': np.random.choice(['Completed', 'Discontinued'], n_patients, p=[0.85, 0.15]),
        'adverse_events_count': np.random.poisson(2, n_patients)
    })
    
    # Add correlated 52-week 6MWD change
    data['week_52_6mwd_change'] = np.where(
        data['treatment_group'] == 'Treatment',
        np.random.normal(20, 15, n_patients),  # Treatment effect
        np.random.normal(-30, 15, n_patients)  # Placebo decline
    )
    
    return data

# Create real data
real_data = create_real_data(n_patients=100)

# Define metadata for SDV
metadata = SingleTableMetadata()

# Detect basic metadata from the dataframe
metadata.detect_from_dataframe(data=real_data)

# Update metadata with specific data types and relationships
metadata.update_column(
    column_name='patient_id',
    sdtype='id'  # Changed to 'id' type
)

metadata.update_column(
    column_name='age',
    sdtype='numerical',
    computer_representation='Float'
)

metadata.update_column(
    column_name='weight_kg',
    sdtype='numerical',
    computer_representation='Float'
)

metadata.update_column(
    column_name='baseline_6mwd',
    sdtype='numerical',
    computer_representation='Float'
)

metadata.update_column(
    column_name='treatment_group',
    sdtype='categorical'
)

metadata.update_column(
    column_name='exon_mutation',
    sdtype='categorical'
)

metadata.update_column(
    column_name='completion_status',
    sdtype='categorical'
)

metadata.update_column(
    column_name='adverse_events_count',
    sdtype='numerical',
    computer_representation='Int64'
)

metadata.update_column(
    column_name='week_52_6mwd_change',
    sdtype='numerical',
    computer_representation='Float'
)

# Set primary key
metadata.set_primary_key('patient_id')

# Create and fit the synthesizer
synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=True
)
synthesizer.fit(real_data)

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=100)

# Validation functions
def plot_distributions(real_df, synthetic_df, columns, filename='distributions.png'):
    """Plot distributions of real vs synthetic data"""
    n_cols = len(columns)
    fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if real_df[col].dtype in ['int64', 'float64']:
            # Numerical columns
            sns.kdeplot(data=real_df, x=col, ax=axes[i], label='Real')
            sns.kdeplot(data=synthetic_df, x=col, ax=axes[i], label='Synthetic')
        else:
            # Categorical columns
            pd.concat([
                real_df[col].value_counts(normalize=True),
                synthetic_df[col].value_counts(normalize=True)
            ], axis=1, keys=['Real', 'Synthetic']).plot(
                kind='bar', ax=axes[i]
            )
        axes[i].set_title(col)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def validate_relationships(real_df, synthetic_df):
    """Validate key relationships in the data"""
    # Treatment effect validation
    real_effect = real_df.groupby('treatment_group')['week_52_6mwd_change'].mean()
    synth_effect = synthetic_df.groupby('treatment_group')['week_52_6mwd_change'].mean()
    
    print("\nTreatment Effect Comparison:")
    print("Real data:")
    print(real_effect)
    print("\nSynthetic data:")
    print(synth_effect)
    
    # Age-weight correlation
    real_corr = real_df['age'].corr(real_df['weight_kg'])
    synth_corr = synthetic_df['age'].corr(synthetic_df['weight_kg'])
    
    print("\nAge-Weight Correlation:")
    print(f"Real data: {real_corr:.3f}")
    print(f"Synthetic data: {synth_corr:.3f}")

# Run validation
columns_to_plot = ['age', 'weight_kg', 'baseline_6mwd', 'week_52_6mwd_change', 
                  'treatment_group', 'exon_mutation']
plot_distributions(real_data, synthetic_data, columns_to_plot)
validate_relationships(real_data, synthetic_data)

# Evaluate quality using SDV's built-in metrics
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

print("\nQuality Report:")
print(quality_report)
# Recheck status on Colab
# Save data
real_data.to_csv('real_dmd_data.csv', index=False)
synthetic_data.to_csv('synthetic_dmd_data.csv', index=False)
