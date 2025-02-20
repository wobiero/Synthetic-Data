import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class DMDTrialDataGenerator:
    def __init__(self, n_patients=100, trial_duration_weeks=52):
        self.n_patients = n_patients
        self.trial_duration_weeks = trial_duration_weeks
        self.assessment_weeks = [0, 12, 24, 36, 48, 52]
        
        # Trial parameters
        self.dropout_rate = 0.15  # 15% dropout rate
        self.treatment_effect = 0.3  # Effect size
        self.placebo_decline = -0.2  # Natural disease progression
        
    def generate_patient_demographics(self):
        """Generate baseline patient characteristics"""
        demographics = pd.DataFrame({
            'patient_id': range(1, self.n_patients + 1),
            'age': np.random.normal(8, 2, self.n_patients).round(1),  # Age 6-10 typical for DMD trials
            'weight_kg': np.random.normal(30, 5, self.n_patients).round(1),
            'height_cm': np.random.normal(125, 10, self.n_patients).round(1),
            'treatment_group': np.random.choice(['Treatment', 'Placebo'], self.n_patients),
            'baseline_6mwd': np.random.normal(350, 50, self.n_patients).round(1),  # 6-minute walk distance
            'baseline_fvc_percent': np.random.normal(85, 10, self.n_patients).round(1),  # Forced vital capacity
            'exon_mutation': np.random.choice([45, 51, 53], self.n_patients, p=[0.4, 0.35, 0.25])  # Common mutations
        })
        
        # Ensure age-appropriate height/weight relationships
        demographics['weight_kg'] = demographics.apply(
            lambda x: max(20, min(50, x['weight_kg'])), axis=1
        )
        return demographics
    
    def generate_6mwd_trajectory(self, baseline, treatment_group, timepoints):
        """Generate 6MWD measurements over time"""
        if treatment_group == 'Treatment':
            effect = self.treatment_effect
        else:
            effect = self.placebo_decline
            
        # Add natural variation and noise
        variation = np.random.normal(0, 0.1, len(timepoints))
        trajectory = baseline * (1 + effect * (timepoints/52) + variation)
        
        # Add more realistic noise based on measurement variability
        noise = np.random.normal(0, 10, len(timepoints))
        trajectory = trajectory + noise
        
        return trajectory.round(1)
    
    def generate_adverse_events(self, n_patients):
        """Generate adverse event data"""
        ae_types = ['Injection site reaction', 'Headache', 'Fatigue', 
                   'Upper respiratory infection', 'Fall', 'Joint pain']
        ae_probabilities = [0.35, 0.25, 0.15, 0.1, 0.1, 0.05]  # Sum = 1.0
        
        adverse_events = []
        for patient_id in range(1, n_patients + 1):
            n_events = np.random.poisson(2)  # Average 2 AEs per patient
            for _ in range(n_events):
                ae_type = np.random.choice(ae_types, p=ae_probabilities)
                start_day = np.random.randint(1, self.trial_duration_weeks * 7)
                duration_days = np.random.randint(1, 14)  # 1-14 days duration
                severity = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.7, 0.25, 0.05])
                
                adverse_events.append({
                    'patient_id': patient_id,
                    'ae_type': ae_type,
                    'start_day': start_day,
                    'duration_days': duration_days,
                    'severity': severity
                })
        
        return pd.DataFrame(adverse_events)
    
    def generate_trial_data(self):
        """Generate complete trial dataset"""
        # Generate demographics
        demographics = self.generate_patient_demographics()
        
        # Generate longitudinal data
        longitudinal_data = []
        for _, patient in demographics.iterrows():
            # Simulate dropout
            dropout_week = np.random.choice(
                [np.inf] + self.assessment_weeks[1:],
                p=[1 - self.dropout_rate] + [self.dropout_rate/5] * 5
            )
            
            # Generate 6MWD measurements
            six_mwd = self.generate_6mwd_trajectory(
                patient['baseline_6mwd'],
                patient['treatment_group'],
                np.array(self.assessment_weeks)
            )
            
            # Generate FVC measurements (similar pattern to 6MWD)
            fvc_trajectory = self.generate_6mwd_trajectory(
                patient['baseline_fvc_percent'],
                patient['treatment_group'],
                np.array(self.assessment_weeks)
            )
            
            # Create longitudinal records
            for week, mwd, fvc in zip(self.assessment_weeks, six_mwd, fvc_trajectory):
                if week <= dropout_week:
                    longitudinal_data.append({
                        'patient_id': patient['patient_id'],
                        'week': week,
                        'six_mwd': mwd,
                        'fvc_percent': fvc,
                        'completed_visit': True
                    })
                else:
                    # Add record showing dropout
                    longitudinal_data.append({
                        'patient_id': patient['patient_id'],
                        'week': week,
                        'six_mwd': None,
                        'fvc_percent': None,
                        'completed_visit': False
                    })
        
        longitudinal_df = pd.DataFrame(longitudinal_data)
        
        # Generate adverse events
        adverse_events = self.generate_adverse_events(self.n_patients)
        
        return {
            'demographics': demographics,
            'longitudinal': longitudinal_df,
            'adverse_events': adverse_events
        }

# Run code
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create generator and generate data
    generator = DMDTrialDataGenerator(n_patients=100, trial_duration_weeks=52)
    trial_data = generator.generate_trial_data()
    
    # Save to CSV files
    trial_data['demographics'].to_csv('dmd_demographics.csv', index=False)
    trial_data['longitudinal'].to_csv('dmd_longitudinal.csv', index=False)
    trial_data['adverse_events'].to_csv('dmd_adverse_events.csv', index=False)
    
    # Print summary statistics
    print("\nDemographic Summary:")
    print(trial_data['demographics'].describe())
    
    print("\nLongitudinal Data Summary:")
    print(trial_data['longitudinal'].groupby('week')['six_mwd'].describe())
    
    print("\nAdverse Events Summary:")
    print(trial_data['adverse_events']['ae_type'].value_counts())