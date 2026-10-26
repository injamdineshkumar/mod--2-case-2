"""
Fuzzy Inference Engine for Medical Diagnosis
"""

import numpy as np


class InferenceEngine:
    """
    Inference engine for fuzzy logic systems
    """
    
    def __init__(self, aggregation_method='max', implication_method='min'):
        """
        Initialize inference engine
        
        Parameters:
        aggregation_method: Method for aggregating rules ('max' or 'sum')
        implication_method: Method for implication ('min' or 'product')
        """
        self.aggregation_method = aggregation_method
        self.implication_method = implication_method
    
    def evaluate_rule(self, fuzzy_values, rule):
        """
        Evaluate a single rule
        
        Parameters:
        fuzzy_values: Dictionary of fuzzy membership values
        rule: Dictionary of symptoms and required membership levels
        
        Returns:
        Rule strength (degree of truth)
        """
        strengths = []
        
        for symptom, required_level in rule.items():
            if symptom in fuzzy_values:
                # Get membership value for the required level
                membership = fuzzy_values[symptom].get(required_level, 0.0)
                strengths.append(membership)
            else:
                strengths.append(0.0)
        
        # Apply AND operator (min)
        rule_strength = min(strengths) if strengths else 0.0
        
        return rule_strength
    
    def aggregate_rules(self, rule_strengths):
        """
        Aggregate multiple rules
        
        Parameters:
        rule_strengths: List of rule strengths
        
        Returns:
        Aggregated strength
        """
        if not rule_strengths:
            return 0.0
        
        if self.aggregation_method == 'max':
            return max(rule_strengths)
        elif self.aggregation_method == 'sum':
            return min(1.0, sum(rule_strengths))
        else:
            return max(rule_strengths)
    
    def apply_implication(self, antecedent_strength, consequent):
        """
        Apply implication operator
        
        Parameters:
        antecedent_strength: Rule strength from antecedent
        consequent: Consequent membership function
        
        Returns:
        Implied consequent
        """
        if self.implication_method == 'min':
            return min(antecedent_strength, consequent)
        elif self.implication_method == 'product':
            return antecedent_strength * consequent
        else:
            return min(antecedent_strength, consequent)
    
    def compose_rules(self, rules, fuzzy_values):
        """
        Compose all rules for a disease
        
        Parameters:
        rules: List of rules
        fuzzy_values: Fuzzy membership values
        
        Returns:
        Disease membership strength
        """
        strengths = []
        
        for rule in rules:
            strength = self.evaluate_rule(fuzzy_values, rule)
            strengths.append(strength)
        
        return self.aggregate_rules(strengths)