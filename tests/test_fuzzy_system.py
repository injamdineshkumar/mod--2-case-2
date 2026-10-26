"""
Unit tests for the fuzzy medical diagnosis system
"""

import unittest
import sys
import os
import pandas as pd
import json
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    MedicalDiagnosisSystem,
    FuzzyMedicalDiagnosisSystem,
    validate_symptoms,
    normalize_symptoms,
    save_diagnosis_results,
    load_patient_data,
    create_symptom_vector,
    get_disease_description,
    get_treatment_recommendations
)
from src.membership_functions import MembershipFunction, SymptomMembership, DiseaseMembership
from src.inference_engine import InferenceEngine


class TestMembershipFunctions(unittest.TestCase):
    """Test cases for membership functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.membership = MembershipFunction()
        self.symptom_membership = SymptomMembership()
        self.disease_membership = DiseaseMembership()
    
    def test_triangular_membership(self):
        """Test triangular membership function"""
        # Test at different points
        self.assertEqual(self.membership.triangular(0, 0, 0.5, 1), 0.0)
        self.assertEqual(self.membership.triangular(0.5, 0, 0.5, 1), 1.0)
        self.assertEqual(self.membership.triangular(1, 0, 0.5, 1), 0.0)
        self.assertEqual(self.membership.triangular(0.25, 0, 0.5, 1), 0.5)
        self.assertEqual(self.membership.triangular(0.75, 0, 0.5, 1), 0.5)
        
        # Test out of bounds
        self.assertEqual(self.membership.triangular(-1, 0, 0.5, 1), 0.0)
        self.assertEqual(self.membership.triangular(2, 0, 0.5, 1), 0.0)
    
    def test_trapezoidal_membership(self):
        """Test trapezoidal membership function"""
        # Test at different points
        self.assertEqual(self.membership.trapezoidal(0, 0, 0.2, 0.8, 1), 0.0)
        self.assertEqual(self.membership.trapezoidal(0.1, 0, 0.2, 0.8, 1), 0.5)
        self.assertEqual(self.membership.trapezoidal(0.5, 0, 0.2, 0.8, 1), 1.0)
        self.assertEqual(self.membership.trapezoidal(0.9, 0, 0.2, 0.8, 1), 0.5)
        self.assertEqual(self.membership.trapezoidal(1, 0, 0.2, 0.8, 1), 0.0)
    
    def test_gaussian_membership(self):
        """Test gaussian membership function"""
        # Test at different points
        self.assertAlmostEqual(self.membership.gaussian(0, 0, 1), 1.0)
        self.assertAlmostEqual(self.membership.gaussian(1, 0, 1), 0.6065, places=3)
        self.assertAlmostEqual(self.membership.gaussian(2, 0, 1), 0.1353, places=3)
        
        # Test with different sigma
        self.assertAlmostEqual(self.membership.gaussian(1, 0, 2), 0.8825, places=3)
    
    def test_symptom_membership(self):
        """Test symptom membership functions"""
        # Test fever membership at various values
        fever_membership = self.symptom_membership.fever_membership(0.5)
        self.assertIn('low', fever_membership)
        self.assertIn('medium', fever_membership)
        self.assertIn('high', fever_membership)
        self.assertGreaterEqual(fever_membership['medium'], 0)
        self.assertLessEqual(fever_membership['medium'], 1)
        
        # Test at extremes
        fever_membership_low = self.symptom_membership.fever_membership(0)
        self.assertEqual(fever_membership_low['low'], 1.0)
        self.assertEqual(fever_membership_low['medium'], 0.0)
        
        fever_membership_high = self.symptom_membership.fever_membership(1)
        self.assertEqual(fever_membership_high['high'], 1.0)
        self.assertEqual(fever_membership_high['medium'], 0.0)
    
    def test_disease_membership(self):
        """Test disease membership functions"""
        # Test influenza membership
        flu_membership = self.disease_membership.get_disease_membership('Influenza', 0.5)
        self.assertIn('low', flu_membership)
        self.assertIn('medium', flu_membership)
        self.assertIn('high', flu_membership)
        
        # Test unknown disease
        unknown = self.disease_membership.get_disease_membership('Unknown', 0.5)
        self.assertIsNone(unknown)


class TestInferenceEngine(unittest.TestCase):
    """Test cases for inference engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = InferenceEngine()
        self.engine_max = InferenceEngine(aggregation_method='max')
        self.engine_sum = InferenceEngine(aggregation_method='sum')
        
        # Sample fuzzy values
        self.fuzzy_values = {
            'Fever': {'low': 0.1, 'medium': 0.7, 'high': 0.3},
            'Cough': {'low': 0.2, 'medium': 0.8, 'high': 0.4},
            'Fatigue': {'low': 0.3, 'medium': 0.6, 'high': 0.2}
        }
    
    def test_evaluate_rule(self):
        """Test rule evaluation"""
        rule = {
            'Fever': 'medium',
            'Cough': 'high'
        }
        
        strength = self.engine.evaluate_rule(self.fuzzy_values, rule)
        self.assertEqual(strength, min(0.7, 0.4))  # Should be 0.4
        
        # Test with missing symptom
        rule_missing = {
            'Headache': 'medium',
            'Fever': 'high'
        }
        strength_missing = self.engine.evaluate_rule(self.fuzzy_values, rule_missing)
        self.assertEqual(strength_missing, 0.0)
    
    def test_aggregate_rules(self):
        """Test rule aggregation"""
        strengths = [0.3, 0.7, 0.5]
        
        # Max aggregation
        aggregated_max = self.engine_max.aggregate_rules(strengths)
        self.assertEqual(aggregated_max, 0.7)
        
        # Sum aggregation
        aggregated_sum = self.engine_sum.aggregate_rules(strengths)
        self.assertEqual(aggregated_sum, min(1.0, 1.5))  # Should be 1.0
        
        # Empty strengths
        self.assertEqual(self.engine.aggregate_rules([]), 0.0)
    
    def test_apply_implication(self):
        """Test implication application"""
        antecedent = 0.7
        consequent = 0.8
        
        # Min implication
        implied = self.engine.apply_implication(antecedent, consequent)
        self.assertEqual(implied, 0.7)
        
        # Product implication
        engine_product = InferenceEngine(implication_method='product')
        implied_product = engine_product.apply_implication(antecedent, consequent)
        self.assertEqual(implied_product, 0.7 * 0.8)
    
    def test_compose_rules(self):
        """Test rule composition"""
        rules = [
            {'Fever': 'high', 'Cough': 'high'},
            {'Fever': 'medium', 'Fatigue': 'high'}
        ]
        
        composed = self.engine.compose_rules(rules, self.fuzzy_values)
        self.assertGreaterEqual(composed, 0)
        self.assertLessEqual(composed, 1)


class TestFuzzySystem(unittest.TestCase):
    """Test cases for fuzzy diagnosis system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = MedicalDiagnosisSystem()
        self.fuzzy_system = FuzzyMedicalDiagnosisSystem()
        
        self.test_patient = {
            'Fever': 0.8,
            'Headache': 0.3,
            'Cough': 0.7,
            'Sore_Throat': 0.2,
            'Fatigue': 0.6,
            'Chest_Pain': 0.1,
            'Shortness_Breath': 0.0
        }
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system.system)
        self.assertIsNotNone(self.fuzzy_system)
        self.assertEqual(len(self.fuzzy_system.diseases), 4)
        self.assertIn('Influenza', self.fuzzy_system.diseases)
        self.assertIn('Common_Cold', self.fuzzy_system.diseases)
        self.assertIn('Pneumonia', self.fuzzy_system.diseases)
        self.assertIn('Chronic_Fatigue', self.fuzzy_system.diseases)
    
    def test_fuzzification(self):
        """Test fuzzification process"""
        fuzzy_values = self.fuzzy_system.fuzzify(self.test_patient)
        
        # Check that all symptoms are processed
        self.assertIn('Fever', fuzzy_values)
        self.assertIn('Cough', fuzzy_values)
        self.assertIn('Fatigue', fuzzy_values)
        
        # Check membership values
        self.assertIn('high', fuzzy_values['Fever'])
        self.assertEqual(fuzzy_values['Fever']['high'], 1.0)
        
        # Check that symptom returns correct membership
        self.assertIn('medium', fuzzy_values['Cough'])
        self.assertIn('high', fuzzy_values['Cough'])
    
    def test_inference(self):
        """Test inference process"""
        fuzzy_values = self.fuzzy_system.fuzzify(self.test_patient)
        disease_probs = self.fuzzy_system.infer(fuzzy_values)
        
        # Check all diseases are present
        for disease in self.fuzzy_system.diseases:
            self.assertIn(disease, disease_probs)
            self.assertGreaterEqual(disease_probs[disease], 0)
            self.assertLessEqual(disease_probs[disease], 1)
        
        # Influenza should have high probability for this patient
        self.assertGreater(disease_probs['Influenza'], 0.3)
    
    def test_defuzzification(self):
        """Test defuzzification process"""
        disease_probs = {
            'Influenza': 0.8,
            'Common_Cold': 0.3,
            'Pneumonia': 0.6,
            'Chronic_Fatigue': 0.1
        }
        
        result = self.fuzzy_system.defuzzify(disease_probs.copy())
        
        # Check primary diagnosis
        self.assertEqual(result['primary_diagnosis'], 'Influenza')
        self.assertAlmostEqual(result['primary_confidence'], 0.8/1.8, places=2)
        
        # Check secondary diagnosis
        self.assertEqual(result['secondary_diagnosis'], 'Pneumonia')
        
        # Check all diseases
        self.assertIn('all_diseases', result)
        self.assertEqual(len(result['all_diseases']), 4)
    
    def test_diagnose(self):
        """Test complete diagnosis pipeline"""
        result = self.system.diagnose_patient(self.test_patient)
        
        # Check required fields
        self.assertIn('primary_diagnosis', result)
        self.assertIn('primary_confidence', result)
        self.assertIn('secondary_diagnosis', result)
        self.assertIn('all_diseases', result)
        self.assertIn('input_symptoms', result)
        self.assertIn('fuzzy_values', result)
        
        # Check valid confidence values
        self.assertGreaterEqual(result['primary_confidence'], 0)
        self.assertLessEqual(result['primary_confidence'], 1)
        
        # Check primary diagnosis is one of the diseases
        self.assertIn(result['primary_diagnosis'], self.fuzzy_system.diseases)
    
    def test_diagnose_invalid_patient(self):
        """Test diagnosis with invalid patient data"""
        # String instead of dict
        result = self.fuzzy_system.diagnose("invalid")
        self.assertIn('error', result)
        
        # Empty dict
        result = self.fuzzy_system.diagnose({})
        # Should still work but with default values
        self.assertIn('primary_diagnosis', result)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, 'test_data.csv')
        
        # Create test data
        test_data = pd.DataFrame({
            'Patient_ID': ['T001', 'T002', 'T003'],
            'Fever': [0.8, 0.3, 0.9],
            'Cough': [0.7, 0.2, 0.8],
            'Fatigue': [0.6, 0.5, 0.7],
            'Headache': [0.3, 0.6, 0.2],
            'Sore_Throat': [0.2, 0.5, 0.3],
            'Chest_Pain': [0.1, 0.0, 0.7],
            'Shortness_Breath': [0.0, 0.0, 0.6],
            'Disease': ['Influenza', 'Common_Cold', 'Pneumonia']
        })
        test_data.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    def test_validate_symptoms(self):
        """Test symptom validation"""
        valid_symptoms = {'Fever': 0.5, 'Cough': 0.3, 'Headache': 0.8}
        invalid_symptoms_type = {'Fever': '0.5', 'Cough': 0.3}
        invalid_symptoms_range = {'Fever': 1.5, 'Cough': -0.1}
        
        self.assertTrue(validate_symptoms(valid_symptoms))
        self.assertFalse(validate_symptoms(invalid_symptoms_type))
        self.assertFalse(validate_symptoms(invalid_symptoms_range))
    
    def test_normalize_symptoms(self):
        """Test symptom normalization"""
        symptoms = {'Fever': 2, 'Cough': 4, 'Headache': 1, 'Fatigue': 3}
        normalized = normalize_symptoms(symptoms)
        
        # Check min-max normalization
        self.assertEqual(normalized['Fever'], (2-1)/(4-1))
        self.assertEqual(normalized['Cough'], 1.0)
        self.assertEqual(normalized['Headache'], 0.0)
        self.assertEqual(normalized['Fatigue'], (3-1)/(4-1))
        
        # Test with single value
        single = {'Fever': 0.5}
        normalized_single = normalize_symptoms(single)
        self.assertEqual(normalized_single['Fever'], 0.5)
    
    def test_save_diagnosis_results(self):
        """Test saving diagnosis results"""
        # Test DataFrame save
        df = pd.DataFrame({
            'Patient_ID': ['T001'],
            'Diagnosis': ['Influenza'],
            'Confidence': [0.85]
        })
        
        csv_path = os.path.join(self.test_dir, 'results.csv')
        save_diagnosis_results(df, csv_path)
        self.assertTrue(os.path.exists(csv_path))
        
        # Test dict save
        dict_results = {'T001': {'diagnosis': 'Influenza', 'confidence': 0.85}}
        json_path = os.path.join(self.test_dir, 'results.json')
        save_diagnosis_results(dict_results, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Test default save
        os.makedirs(os.path.join(self.test_dir, 'outputs'), exist_ok=True)
        save_diagnosis_results(dict_results, os.path.join(self.test_dir, 'outputs'))
        # Should create timestamped file
        files = os.listdir(os.path.join(self.test_dir, 'outputs'))
        self.assertTrue(any(f.startswith('diagnosis_results_') for f in files))
    
    def test_load_patient_data(self):
        """Test loading patient data"""
        df = load_patient_data(self.test_csv_path)
        
        self.assertEqual(len(df), 3)
        self.assertIn('Patient_ID', df.columns)
        self.assertIn('Fever', df.columns)
        self.assertIn('Disease', df.columns)
        
        # Test file not found
        with self.assertRaises(FileNotFoundError):
            load_patient_data('nonexistent_file.csv')
    
    def test_create_symptom_vector(self):
        """Test symptom vector creation"""
        # Test with dict
        symptoms_dict = {'Fever': 0.8, 'Cough': 0.7, 'Headache': 0.3}
        vector = create_symptom_vector(symptoms_dict)
        
        self.assertIn('Fever', vector)
        self.assertIn('Headache', vector)
        self.assertIn('Shortness_Breath', vector)  # Should be filled with default
        self.assertEqual(vector['Shortness_Breath'], 0.0)
        
        # Test with pandas Series
        df = pd.read_csv(self.test_csv_path)
        row = df.iloc[0]
        vector_series = create_symptom_vector(row)
        
        self.assertIn('Fever', vector_series)
        self.assertIn('Cough', vector_series)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            create_symptom_vector("invalid")
    
    def test_get_disease_description(self):
        """Test getting disease descriptions"""
        descriptions = {
            'Influenza': 'Viral infection affecting respiratory system',
            'Common_Cold': 'Mild viral infection of upper respiratory tract',
            'Pneumonia': 'Inflammation of lungs often caused by infection',
            'Chronic_Fatigue': 'Persistent fatigue lasting at least 6 months',
            'Unknown': 'Unable to determine specific diagnosis'
        }
        
        for disease, expected_desc in descriptions.items():
            description = get_disease_description(disease)
            self.assertIn(expected_desc, description)
    
    def test_get_treatment_recommendations(self):
        """Test getting treatment recommendations"""
        treatments = {
            'Influenza': ['Rest and hydration', 'Over-the-counter fever reducers'],
            'Common_Cold': ['Rest and hydration', 'Over-the-counter cold medications'],
            'Pneumonia': ['Seek immediate medical attention', 'Antibiotics (if bacterial)'],
            'Chronic_Fatigue': ['Consult healthcare provider for diagnosis', 'Gradual exercise program']
        }
        
        for disease, expected_treatments in treatments.items():
            recommendations = get_treatment_recommendations(disease)
            for rec in expected_treatments:
                self.assertIn(rec, recommendations)
        
        # Test unknown disease
        unknown_treatments = get_treatment_recommendations('Unknown')
        self.assertIn('Consult healthcare provider for proper treatment', unknown_treatments)


class TestMedicalDiagnosisSystem(unittest.TestCase):
    """Integration tests for medical diagnosis system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = MedicalDiagnosisSystem()
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, 'test_data.csv')
        
        # Create test data
        test_data = pd.DataFrame({
            'Patient_ID': ['T001', 'T002', 'T003'],
            'Fever': [0.8, 0.3, 0.9],
            'Cough': [0.7, 0.2, 0.8],
            'Fatigue': [0.6, 0.5, 0.7],
            'Headache': [0.3, 0.6, 0.2],
            'Sore_Throat': [0.2, 0.5, 0.3],
            'Chest_Pain': [0.1, 0.0, 0.7],
            'Shortness_Breath': [0.0, 0.0, 0.6],
            'Disease': ['Influenza', 'Common_Cold', 'Pneumonia']
        })
        test_data.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    def test_diagnose_patient_influenza(self):
        """Test diagnosis for influenza patient"""
        symptoms = {
            'Fever': 0.8,
            'Headache': 0.3,
            'Cough': 0.7,
            'Sore_Throat': 0.2,
            'Fatigue': 0.6,
            'Chest_Pain': 0.1,
            'Shortness_Breath': 0.0
        }
        
        result = self.system.diagnose_patient(symptoms)
        
        # Should identify influenza
        self.assertEqual(result['primary_diagnosis'], 'Influenza')
        self.assertGreater(result['primary_confidence'], 0.3)
    
    def test_diagnose_patient_pneumonia(self):
        """Test diagnosis for pneumonia patient"""
        symptoms = {
            'Fever': 0.9,
            'Headache': 0.2,
            'Cough': 0.8,
            'Sore_Throat': 0.3,
            'Fatigue': 0.7,
            'Chest_Pain': 0.7,
            'Shortness_Breath': 0.6
        }
        
        result = self.system.diagnose_patient(symptoms)
        
        # Should identify pneumonia
        self.assertEqual(result['primary_diagnosis'], 'Pneumonia')
        self.assertGreater(result['primary_confidence'], 0.3)
    
    def test_diagnose_patient_chronic_fatigue(self):
        """Test diagnosis for chronic fatigue patient"""
        symptoms = {
            'Fever': 0.2,
            'Headache': 0.7,
            'Cough': 0.1,
            'Sore_Throat': 0.4,
            'Fatigue': 0.8,
            'Chest_Pain': 0.0,
            'Shortness_Breath': 0.0
        }
        
        result = self.system.diagnose_patient(symptoms)
        
        # Should identify chronic fatigue
        self.assertEqual(result['primary_diagnosis'], 'Chronic_Fatigue')
        self.assertGreater(result['primary_confidence'], 0.3)
    
    def test_diagnose_from_csv(self):
        """Test batch diagnosis from CSV"""
        results = self.system.diagnose_from_csv(self.test_csv_path)
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertIn('Patient_ID', results.columns)
        self.assertIn('Actual_Disease', results.columns)
        self.assertIn('Diagnosed_Disease', results.columns)
        self.assertIn('Confidence', results.columns)
        
        # Check that results have valid values
        self.assertTrue((results['Confidence'] >= 0).all())
        self.assertTrue((results['Confidence'] <= 1).all())
        
        # Check first patient
        first_result = results.iloc[0]
        self.assertEqual(first_result['Patient_ID'], 'T001')
        self.assertEqual(first_result['Actual_Disease'], 'Influenza')
    
    def test_system_rules(self):
        """Test system rules are properly defined"""
        rules = self.system.system.rules
        
        # Check all diseases have rules
        for disease in ['Influenza', 'Common_Cold', 'Pneumonia', 'Chronic_Fatigue']:
            self.assertIn(disease, rules)
            self.assertIsInstance(rules[disease], dict)
            self.assertGreater(len(rules[disease]), 0)
        
        # Check specific rule structures
        self.assertIn('Fever', rules['Influenza'])
        self.assertIn('Sore_Throat', rules['Common_Cold'])
        self.assertIn('Chest_Pain', rules['Pneumonia'])
        self.assertIn('Fatigue', rules['Chronic_Fatigue'])
    
    def test_symptom_mapping(self):
        """Test symptom mapping in fuzzy system"""
        symptom_mapping = self.system.system.symptom_mapping
        
        expected_symptoms = ['Fever', 'Headache', 'Cough', 'Sore_Throat', 
                            'Fatigue', 'Chest_Pain', 'Shortness_Breath']
        
        for symptom in expected_symptoms:
            self.assertIn(symptom, symptom_mapping)
            self.assertTrue(callable(symptom_mapping[symptom]))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)