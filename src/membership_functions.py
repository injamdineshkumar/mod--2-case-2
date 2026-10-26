"""
Membership Functions for Fuzzy Medical Diagnosis System
"""

import numpy as np

class MembershipFunction:
    """Base class for membership functions"""
    
    @staticmethod
    def triangular(x, a, b, c):
        """
        Triangular membership function
        
        Parameters:
        x: Input value
        a: Left bound
        b: Peak (center)
        c: Right bound
        
        Returns:
        Membership degree (0-1)
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        return 0.0
    
    @staticmethod
    def trapezoidal(x, a, b, c, d):
        """
        Trapezoidal membership function
        
        Parameters:
        x: Input value
        a: Left foot
        b: Left shoulder
        c: Right shoulder
        d: Right foot
        
        Returns:
        Membership degree (0-1)
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
        return 0.0
    
    @staticmethod
    def gaussian(x, mean, sigma):
        """
        Gaussian membership function
        
        Parameters:
        x: Input value
        mean: Center of gaussian
        sigma: Spread of gaussian
        
        Returns:
        Membership degree (0-1)
        """
        return np.exp(-((x - mean) ** 2) / (2 * (sigma ** 2)))


class SymptomMembership:
    """Define membership functions for each symptom"""
    
    def __init__(self):
        self.membership_func = MembershipFunction()
    
    def fever_membership(self, value):
        """Fever symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}
    
    def headache_membership(self, value):
        """Headache symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}
    
    def cough_membership(self, value):
        """Cough symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}
    
    def sore_throat_membership(self, value):
        """Sore throat symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}
    
    def fatigue_membership(self, value):
        """Fatigue symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}
    
    def chest_pain_membership(self, value):
        """Chest pain symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}
    
    def shortness_breath_membership(self, value):
        """Shortness of breath symptom: Low, Medium, High"""
        low = self.membership_func.triangular(value, 0, 0, 0.4)
        medium = self.membership_func.triangular(value, 0.2, 0.5, 0.8)
        high = self.membership_func.triangular(value, 0.6, 1.0, 1.0)
        return {'low': low, 'medium': medium, 'high': high}


class DiseaseMembership:
    """Define membership functions for diseases"""
    
    def __init__(self):
        self.membership_func = MembershipFunction()
    
    def get_disease_membership(self, disease, value):
        """Get membership value for a specific disease"""
        disease_ranges = {
            'Influenza': {'low': (0, 0, 0.3), 'medium': (0.2, 0.5, 0.7), 'high': (0.6, 1.0, 1.0)},
            'Common_Cold': {'low': (0, 0, 0.3), 'medium': (0.2, 0.5, 0.7), 'high': (0.6, 1.0, 1.0)},
            'Pneumonia': {'low': (0, 0, 0.3), 'medium': (0.2, 0.5, 0.7), 'high': (0.6, 1.0, 1.0)},
            'Chronic_Fatigue': {'low': (0, 0, 0.3), 'medium': (0.2, 0.5, 0.7), 'high': (0.6, 1.0, 1.0)}
        }
        
        if disease in disease_ranges:
            ranges = disease_ranges[disease]
            membership = {}
            for level, (a, b, c) in ranges.items():
                membership[level] = self.membership_func.triangular(value, a, b, c)
            return membership
        return None