"""
Utility functions for the fuzzy medical diagnosis system
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def validate_symptoms(symptoms):
    """
    Validate symptom values are between 0 and 1
    
    Parameters:
    symptoms: Dictionary of symptom values
    
    Returns:
    Boolean indicating if validation passed
    """
    for symptom, value in symptoms.items():
        if not isinstance(value, (int, float)):
            return False
        if value < 0 or value > 1:
            return False
    return True


def normalize_symptoms(symptoms, method='min_max'):
    """
    Normalize symptom values
    
    Parameters:
    symptoms: Dictionary of symptom values
    method: Normalization method ('min_max' or 'z_score')
    
    Returns:
    Normalized symptom dictionary
    """
    if method == 'min_max':
        values = list(symptoms.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in symptoms.keys()}
        
        normalized = {}
        for k, v in symptoms.items():
            normalized[k] = (v - min_val) / (max_val - min_val)
        return normalized
    
    return symptoms


def save_diagnosis_results(results, output_path):
    """
    Save diagnosis results to file
    
    Parameters:
    results: Diagnosis results
    output_path: Path to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if isinstance(results, pd.DataFrame):
        results.to_csv(output_path, index=False)
    elif isinstance(results, dict):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Save as JSON with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"diagnosis_results_{timestamp}.json"
        full_path = os.path.join(output_path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)


def load_patient_data(csv_path):
    """
    Load patient symptom data from CSV
    
    Parameters:
    csv_path: Path to CSV file
    
    Returns:
    DataFrame with patient data
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def create_symptom_vector(symptoms):
    """
    Create a standardized symptom vector
    
    Parameters:
    symptoms: Dictionary or DataFrame row
    
    Returns:
    Dictionary with standardized symptom values
    """
    standard_symptoms = [
        'Fever', 'Headache', 'Cough', 'Sore_Throat', 
        'Fatigue', 'Chest_Pain', 'Shortness_Breath'
    ]
    
    if isinstance(symptoms, pd.Series):
        # Convert row to dictionary
        symptom_dict = {}
        for sym in standard_symptoms:
            if sym in symptoms.index:
                symptom_dict[sym] = float(symptoms[sym])
        return symptom_dict
    elif isinstance(symptoms, dict):
        # Ensure all standard symptoms are present
        result = {}
        for sym in standard_symptoms:
            result[sym] = symptoms.get(sym, 0.0)
        return result
    else:
        raise ValueError("Symptoms must be dictionary or pandas Series")


def get_disease_description(disease):
    """
    Get description of a disease
    
    Parameters:
    disease: Disease name
    
    Returns:
    Description string
    """
    descriptions = {
        'Influenza': 'Viral infection affecting respiratory system. Symptoms include fever, cough, body aches.',
        'Common_Cold': 'Mild viral infection of upper respiratory tract. Symptoms include sore throat, runny nose.',
        'Pneumonia': 'Inflammation of lungs often caused by infection. Symptoms include fever, cough, chest pain.',
        'Chronic_Fatigue': 'Persistent fatigue lasting at least 6 months. Symptoms include extreme tiredness, headaches.',
        'Unknown': 'Unable to determine specific diagnosis.'
    }
    
    return descriptions.get(disease, 'Description not available.')


def get_treatment_recommendations(disease):
    """
    Get treatment recommendations for a disease
    
    Parameters:
    disease: Disease name
    
    Returns:
    List of treatment recommendations
    """
    treatments = {
        'Influenza': [
            'Rest and hydration',
            'Over-the-counter fever reducers',
            'Antiviral medications if within 48 hours of symptoms',
            'Consult healthcare provider if symptoms severe'
        ],
        'Common_Cold': [
            'Rest and hydration',
            'Over-the-counter cold medications',
            'Salt water gargle for sore throat',
            'Use humidifier for congestion'
        ],
        'Pneumonia': [
            'Seek immediate medical attention',
            'Antibiotics (if bacterial)',
            'Rest and hydration',
            'Follow prescribed treatment plan exactly'
        ],
        'Chronic_Fatigue': [
            'Consult healthcare provider for diagnosis',
            'Gradual exercise program',
            'Sleep hygiene improvement',
            'Stress management techniques',
            'Cognitive behavioral therapy'
        ]
    }
    
    return treatments.get(disease, ['Consult healthcare provider for proper treatment'])