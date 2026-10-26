"""
Main script for Fuzzy Logic Medical Diagnosis System
"""

import pandas as pd
import json
from src import (
    MedicalDiagnosisSystem,
    load_patient_data,
    save_diagnosis_results,
    get_disease_description,
    get_treatment_recommendations
)


def print_diagnosis(diagnosis):
    """
    Pretty print diagnosis results
    """
    print("\n" + "="*60)
    print("DIAGNOSIS RESULTS")
    print("="*60)
    
    print(f"\nPrimary Diagnosis: {diagnosis['primary_diagnosis']}")
    print(f"Confidence Level: {diagnosis['primary_confidence']:.2%}")
    
    if diagnosis['secondary_diagnosis'] != 'None':
        print(f"\nSecondary Diagnosis: {diagnosis['secondary_diagnosis']}")
        print(f"Confidence Level: {diagnosis['secondary_confidence']:.2%}")
    
    print("\n" + "-"*60)
    print("DISEASE PROBABILITIES")
    print("-"*60)
    
    for disease, prob in diagnosis['all_diseases'].items():
        print(f"{disease:20s}: {prob:.2%}")
    
    print("\n" + "-"*60)
    print("SYMPTOMS ANALYZED")
    print("-"*60)
    
    for symptom, value in diagnosis['input_symptoms'].items():
        print(f"{symptom:20s}: {value:.2f}")
    
    # Get disease description and treatment
    disease = diagnosis['primary_diagnosis']
    print("\n" + "="*60)
    print("ADDITIONAL INFORMATION")
    print("="*60)
    
    print(f"\nDisease Description:")
    print(get_disease_description(disease))
    
    print(f"\nTreatment Recommendations:")
    recommendations = get_treatment_recommendations(disease)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n" + "="*60)


def demo_single_patient():
    """
    Demonstrate diagnosis for a single patient
    """
    print("\n" + "="*60)
    print("FUZZY LOGIC MEDICAL DIAGNOSIS SYSTEM")
    print("="*60)
    
    # Create diagnosis system
    system = MedicalDiagnosisSystem()
    
    # Example patient symptoms
    patient_symptoms = {
        'Fever': 0.8,
        'Headache': 0.3,
        'Cough': 0.7,
        'Sore_Throat': 0.2,
        'Fatigue': 0.6,
        'Chest_Pain': 0.1,
        'Shortness_Breath': 0.0
    }
    
    print("\nPatient Symptoms:")
    for symptom, value in patient_symptoms.items():
        print(f"  {symptom:20s}: {value:.2f}")
    
    # Perform diagnosis
    diagnosis = system.diagnose_patient(patient_symptoms)
    
    # Print results
    print_diagnosis(diagnosis)
    
    return diagnosis


def demo_batch_diagnosis():
    """
    Demonstrate batch diagnosis from dataset
    """
    print("\n" + "="*60)
    print("BATCH DIAGNOSIS FROM DATASET")
    print("="*60)
    
    # Load dataset
    data_path = 'data/symptoms_dataset.csv'
    df = load_patient_data(data_path)
    
    print(f"\nLoaded {len(df)} patient records")
    print("\nSample of data:")
    print(df.head(10))
    
    # Create system and diagnose
    system = MedicalDiagnosisSystem()
    results = system.diagnose_from_csv(data_path)
    
    # Calculate accuracy
    correct = sum(results['Actual_Disease'] == results['Diagnosed_Disease'])
    accuracy = correct / len(results)
    
    print("\n" + "-"*60)
    print("DIAGNOSIS RESULTS SUMMARY")
    print("-"*60)
    print(f"Total patients: {len(results)}")
    print(f"Correct diagnoses: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Show confusion matrix
    print("\nConfusion Matrix:")
    print("="*60)
    confusion = pd.crosstab(results['Actual_Disease'], results['Diagnosed_Disease'])
    print(confusion)
    
    # Show sample results
    print("\nSample Diagnosis Results:")
    print(results[['Patient_ID', 'Actual_Disease', 'Diagnosed_Disease', 'Confidence']].head(10))
    
    # Save results
    save_path = 'outputs/diagnosis_results.csv'
    save_diagnosis_results(results, save_path)
    print(f"\nResults saved to: {save_path}")
    
    return results


def interactive_diagnosis():
    """
    Interactive diagnosis mode
    """
    print("\n" + "="*60)
    print("INTERACTIVE DIAGNOSIS MODE")
    print("="*60)
    print("\nEnter symptom values (0-1) or 'q' to quit")
    
    system = MedicalDiagnosisSystem()
    symptoms = {
        'Fever': 0.0,
        'Headache': 0.0,
        'Cough': 0.0,
        'Sore_Throat': 0.0,
        'Fatigue': 0.0,
        'Chest_Pain': 0.0,
        'Shortness_Breath': 0.0
    }
    
    while True:
        print("\nSymptom List:")
        for i, (symptom, value) in enumerate(symptoms.items(), 1):
            print(f"{i}. {symptom:20s}: {value:.2f}")
        
        choice = input("\nEnter symptom number to modify (or 'd' for diagnosis, 'q' to quit): ")
        
        if choice.lower() == 'q':
            break
        elif choice.lower() == 'd':
            if any(v > 0 for v in symptoms.values()):
                diagnosis = system.diagnose_patient(symptoms)
                print_diagnosis(diagnosis)
            else:
                print("Please enter some symptoms first.")
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(symptoms):
                symptom = list(symptoms.keys())[idx]
                value = float(input(f"Enter value for {symptom} (0-1): "))
                if 0 <= value <= 1:
                    symptoms[symptom] = value
                else:
                    print("Value must be between 0 and 1.")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """
    Main function to run the diagnosis system
    """
    print("FUZZY LOGIC MEDICAL DIAGNOSIS SYSTEM")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Single Patient Diagnosis")
        print("2. Batch Diagnosis from Dataset")
        print("3. Interactive Diagnosis Mode")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            demo_single_patient()
        elif choice == '2':
            demo_batch_diagnosis()
        elif choice == '3':
            interactive_diagnosis()
        elif choice == '4':
            print("\nThank you for using the diagnosis system!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()