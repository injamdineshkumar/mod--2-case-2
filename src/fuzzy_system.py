"""
Fuzzy Inference System for Medical Diagnosis
"""

import pandas as pd
import numpy as np
from .membership_functions import SymptomMembership, DiseaseMembership
from .inference_engine import InferenceEngine


class FuzzyMedicalDiagnosisSystem:
    """
    Fuzzy Logic based medical diagnosis system
    """
    
    def __init__(self):
        self.symptom_membership = SymptomMembership()
        self.disease_membership = DiseaseMembership()
        self.inference_engine = InferenceEngine()
        self.symptom_mapping = {
            'Fever': self.symptom_membership.fever_membership,
            'Headache': self.symptom_membership.headache_membership,
            'Cough': self.symptom_membership.cough_membership,
            'Sore_Throat': self.symptom_membership.sore_throat_membership,
            'Fatigue': self.symptom_membership.fatigue_membership,
            'Chest_Pain': self.symptom_membership.chest_pain_membership,
            'Shortness_Breath': self.symptom_membership.shortness_breath_membership
        }
        self.diseases = ['Influenza', 'Common_Cold', 'Pneumonia', 'Chronic_Fatigue']
        
        # Define fuzzy rules for each disease
        self.rules = self._define_rules()
    
    def _define_rules(self):
        """
        Define fuzzy rules for disease diagnosis
        
        Returns:
        Dictionary of rules for each disease
        """
        rules = {
            'Influenza': {
                'Fever': 'high',
                'Cough': 'medium',
                'Fatigue': 'medium',
                'Headache': 'low'
            },
            'Common_Cold': {
                'Sore_Throat': 'high',
                'Cough': 'low',
                'Headache': 'medium',
                'Fatigue': 'low'
            },
            'Pneumonia': {
                'Fever': 'high',
                'Cough': 'high',
                'Chest_Pain': 'medium',
                'Shortness_Breath': 'medium',
                'Fatigue': 'high'
            },
            'Chronic_Fatigue': {
                'Fatigue': 'high',
                'Headache': 'medium',
                'Sore_Throat': 'low'
            }
        }
        return rules
    
    def fuzzify(self, symptoms):
        """
        Convert crisp symptom values to fuzzy membership values
        
        Parameters:
        symptoms: Dictionary of symptom values (0-1)
        
        Returns:
        Dictionary of fuzzy membership values for each symptom
        """
        fuzzy_values = {}
        
        for symptom, value in symptoms.items():
            if symptom in self.symptom_mapping:
                fuzzy_values[symptom] = self.symptom_mapping[symptom](value)
        
        return fuzzy_values
    
    def infer(self, fuzzy_values):
        """
        Apply inference engine to determine disease probabilities
        
        Parameters:
        fuzzy_values: Fuzzy membership values for symptoms
        
        Returns:
        Dictionary of disease probabilities
        """
        disease_probabilities = {}
        
        for disease in self.diseases:
            if disease in self.rules:
                rule = self.rules[disease]
                # Calculate rule strength
                strength = self.inference_engine.evaluate_rule(fuzzy_values, rule)
                disease_probabilities[disease] = strength
        
        return disease_probabilities
    
    def defuzzify(self, disease_probabilities):
        """
        Convert fuzzy disease probabilities to crisp diagnosis levels
        
        Parameters:
        disease_probabilities: Dictionary of disease probabilities
        
        Returns:
        Dictionary with diagnosis levels and confidence
        """
        # Normalize probabilities
        total = sum(disease_probabilities.values())
        if total > 0:
            for disease in disease_probabilities:
                disease_probabilities[disease] /= total
        
        # Determine primary and secondary diagnoses
        sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        diagnosis = {
            'primary_diagnosis': sorted_diseases[0][0] if sorted_diseases else 'Unknown',
            'primary_confidence': sorted_diseases[0][1] if sorted_diseases else 0.0,
            'secondary_diagnosis': sorted_diseases[1][0] if len(sorted_diseases) > 1 else 'None',
            'secondary_confidence': sorted_diseases[1][1] if len(sorted_diseases) > 1 else 0.0,
            'all_diseases': dict(sorted_diseases)
        }
        
        return diagnosis
    
    def diagnose(self, patient_data):
        """
        Complete diagnosis pipeline
        
        Parameters:
        patient_data: Dictionary with symptom values or patient ID
        
        Returns:
        Diagnosis results
        """
        # If patient ID provided, fetch from dataset
        if isinstance(patient_data, str) or isinstance(patient_data, int):
            # This would typically load from database
            return {"error": "Please provide symptom values"}
        
        # Step 1: Fuzzification
        fuzzy_values = self.fuzzify(patient_data)
        
        # Step 2: Inference
        disease_probabilities = self.infer(fuzzy_values)
        
        # Step 3: Defuzzification
        diagnosis_result = self.defuzzify(disease_probabilities)
        
        # Add input symptoms to results
        diagnosis_result['input_symptoms'] = patient_data
        diagnosis_result['fuzzy_values'] = fuzzy_values
        
        return diagnosis_result
    
    def diagnose_batch(self, symptoms_list):
        """
        Diagnose multiple patients
        
        Parameters:
        symptoms_list: List of symptom dictionaries
        
        Returns:
        List of diagnosis results
        """
        results = []
        for symptoms in symptoms_list:
            result = self.diagnose(symptoms)
            results.append(result)
        return results


class MedicalDiagnosisSystem:
    """
    Wrapper class for easy use of the fuzzy diagnosis system
    """
    
    def __init__(self):
        self.system = FuzzyMedicalDiagnosisSystem()
        self.disease_labels = ['Influenza', 'Common_Cold', 'Pneumonia', 'Chronic_Fatigue']
    
    def diagnose_patient(self, symptoms):
        """
        Diagnose a single patient
        
        Parameters:
        symptoms: Dict with symptom names as keys and values 0-1
        
        Example:
        symptoms = {
            'Fever': 0.8,
            'Cough': 0.7,
            'Fatigue': 0.6,
            'Headache': 0.3
        }
        """
        result = self.system.diagnose(symptoms)
        return result
    
    def diagnose_from_csv(self, csv_path):
        """
        Diagnose patients from CSV file
        
        Parameters:
        csv_path: Path to CSV file with symptom data
        
        Returns:
        DataFrame with diagnosis results
        """
        df = pd.read_csv(csv_path)
        
        # Extract symptoms (exclude Patient_ID and Disease columns)
        symptom_cols = [col for col in df.columns if col not in ['Patient_ID', 'Disease']]
        
        results = []
        for _, row in df.iterrows():
            symptoms = {col: row[col] for col in symptom_cols}
            diagnosis = self.diagnose_patient(symptoms)
            results.append({
                'Patient_ID': row['Patient_ID'] if 'Patient_ID' in df.columns else f"P{_}",
                'Actual_Disease': row['Disease'] if 'Disease' in df.columns else 'Unknown',
                'Diagnosed_Disease': diagnosis['primary_diagnosis'],
                'Confidence': diagnosis['primary_confidence'],
                'Secondary_Diagnosis': diagnosis['secondary_diagnosis'],
                'Secondary_Confidence': diagnosis['secondary_confidence']
            })
        
        return pd.DataFrame(results)