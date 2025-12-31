"""
Decision Engine - Autonomous decision-making system for G.O.A.T

Implements the core decision-making logic that enables G.O.A.T to:
- Analyze situations and context
- Evaluate multiple options
- Make autonomous choices
- Learn from outcomes
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Decision:
    """Represents a single decision made by G.O.A.T"""
    timestamp: str
    context: str
    options: List[str]
    chosen_option: str
    confidence: float
    reasoning: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    success: Optional[bool] = None


class DecisionEngine:
    """
    Advanced decision-making engine with learning capabilities.
    
    The DecisionEngine analyzes context, evaluates options, and makes
    autonomous decisions based on learned patterns and explicit goals.
    """
    
    def __init__(self, decision_log_path: str = "decision_history.json"):
        """
        Initialize the decision engine.
        
        Args:
            decision_log_path: Path to store decision history
        """
        self.decision_log_path = decision_log_path
        self.decision_history: List[Decision] = []
        self.decision_weights: Dict[str, float] = {}
        self.learning_rate = 0.1
        self.confidence_threshold = 0.75
        
        self._load_history()
    
    def make_decision(
        self,
        context: str,
        options: List[str],
        criteria: Optional[Dict[str, float]] = None
    ) -> Decision:
        """
        Make an autonomous decision given context and options.
        
        Args:
            context: Description of the situation requiring a decision
            options: List of possible actions/choices
            criteria: Optional weights for decision criteria
            
        Returns:
            Decision object containing the choice and reasoning
        """
        if not options:
            raise ValueError("Must provide at least one option")
        
        # Analyze each option
        scores = self._evaluate_options(context, options, criteria)
        
        # Select best option
        best_idx = np.argmax(scores)
        chosen_option = options[best_idx]
        confidence = float(scores[best_idx])
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            context, options, scores, best_idx, criteria
        )
        
        # Predict expected outcome
        expected_outcome = self._predict_outcome(context, chosen_option)
        
        # Create decision record
        decision = Decision(
            timestamp=datetime.now().isoformat(),
            context=context,
            options=options,
            chosen_option=chosen_option,
            confidence=confidence,
            reasoning=reasoning,
            expected_outcome=expected_outcome
        )
        
        # Log decision
        self.decision_history.append(decision)
        self._save_decision(decision)
        
        return decision
    
    def _evaluate_options(
        self,
        context: str,
        options: List[str],
        criteria: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """
        Evaluate each option and return scores.
        """
        scores = np.zeros(len(options))
        
        for idx, option in enumerate(options):
            # Base score from historical success
            historical_score = self._get_historical_score(context, option)
            
            # Criteria-based scoring
            criteria_score = self._apply_criteria(option, criteria) if criteria else 0.5
            
            # Context relevance
            relevance_score = self._compute_relevance(context, option)
            
            # Risk assessment
            risk_penalty = self._assess_risk(option)
            
            # Combine scores
            scores[idx] = (
                0.3 * historical_score +
                0.3 * criteria_score +
                0.3 * relevance_score -
                0.1 * risk_penalty
            )
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _get_historical_score(self, context: str, option: str) -> float:
        """
        Calculate score based on historical decision outcomes.
        """
        relevant_decisions = [
            d for d in self.decision_history
            if d.success is not None and option in d.chosen_option
        ]
        
        if not relevant_decisions:
            return 0.5  # Neutral score for unknown options
        
        successes = sum(1 for d in relevant_decisions if d.success)
        return successes / len(relevant_decisions)
    
    def _apply_criteria(self, option: str, criteria: Dict[str, float]) -> float:
        """
        Score option based on provided criteria.
        """
        score = 0.0
        total_weight = sum(criteria.values())
        
        for criterion, weight in criteria.items():
            # Simple keyword matching for demonstration
            if criterion.lower() in option.lower():
                score += weight / total_weight
        
        return score
    
    def _compute_relevance(self, context: str, option: str) -> float:
        """
        Compute how relevant an option is to the context.
        """
        # Simple word overlap for demonstration
        context_words = set(context.lower().split())
        option_words = set(option.lower().split())
        
        if not context_words:
            return 0.5
        
        overlap = len(context_words & option_words)
        return min(1.0, overlap / len(context_words))
    
    def _assess_risk(self, option: str) -> float:
        """
        Assess risk level of an option.
        """
        risk_keywords = ['delete', 'remove', 'permanent', 'irreversible', 'critical']
        risk_score = sum(1 for keyword in risk_keywords if keyword in option.lower())
        return min(1.0, risk_score * 0.2)
    
    def _generate_reasoning(
        self,
        context: str,
        options: List[str],
        scores: np.ndarray,
        chosen_idx: int,
        criteria: Optional[Dict[str, float]]
    ) -> str:
        """
        Generate human-readable reasoning for the decision.
        """
        chosen_option = options[chosen_idx]
        chosen_score = scores[chosen_idx]
        
        reasoning_parts = [
            f"Given the context: '{context}'",
            f"Evaluated {len(options)} possible options.",
            f"Selected '{chosen_option}' with confidence {chosen_score:.2%}."
        ]
        
        if criteria:
            reasoning_parts.append(
                f"Decision weighted by criteria: {', '.join(criteria.keys())}."
            )
        
        # Compare with next best option
        if len(options) > 1:
            sorted_indices = np.argsort(scores)[::-1]
            second_best_idx = sorted_indices[1]
            margin = scores[chosen_idx] - scores[second_best_idx]
            reasoning_parts.append(
                f"Margin over next best option: {margin:.2%}."
            )
        
        return " ".join(reasoning_parts)
    
    def _predict_outcome(self, context: str, option: str) -> str:
        """
        Predict the expected outcome of choosing this option.
        """
        # Simple outcome prediction based on context
        predictions = [
            f"Expected to address the situation described in: {context[:50]}...",
            f"Action '{option}' should lead to resolution.",
            "Monitoring for actual outcome to validate prediction."
        ]
        return " ".join(predictions)
    
    def record_outcome(self, decision: Decision, outcome: str, success: bool):
        """
        Record the actual outcome of a decision for learning.
        
        Args:
            decision: The decision to update
            outcome: Description of what actually happened
            success: Whether the decision achieved its goal
        """
        decision.actual_outcome = outcome
        decision.success = success
        
        # Update weights based on outcome
        self._update_weights(decision)
        
        # Save updated decision
        self._save_decision(decision)
    
    def _update_weights(self, decision: Decision):
        """
        Update decision weights based on outcome.
        """
        option_key = decision.chosen_option
        
        if option_key not in self.decision_weights:
            self.decision_weights[option_key] = 0.5
        
        # Adjust weight based on success
        adjustment = self.learning_rate if decision.success else -self.learning_rate
        self.decision_weights[option_key] = np.clip(
            self.decision_weights[option_key] + adjustment,
            0.0, 1.0
        )
    
    def _load_history(self):
        """
        Load decision history from disk.
        """
        try:
            with open(self.decision_log_path, 'r') as f:
                data = json.load(f)
                self.decision_history = [Decision(**d) for d in data]
        except FileNotFoundError:
            self.decision_history = []
    
    def _save_decision(self, decision: Decision):
        """
        Save a decision to the log file.
        """
        try:
            # Load existing decisions
            try:
                with open(self.decision_log_path, 'r') as f:
                    existing = json.load(f)
            except FileNotFoundError:
                existing = []
            
            # Add or update decision
            decision_dict = asdict(decision)
            existing.append(decision_dict)
            
            # Save back to file
            with open(self.decision_log_path, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save decision: {e}")
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """
        Get statistics about decision-making performance.
        
        Returns:
            Dictionary with various decision metrics
        """
        total_decisions = len(self.decision_history)
        decisions_with_outcome = [d for d in self.decision_history if d.success is not None]
        
        if not decisions_with_outcome:
            success_rate = 0.0
        else:
            success_rate = sum(d.success for d in decisions_with_outcome) / len(decisions_with_outcome)
        
        avg_confidence = np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0
        
        return {
            'total_decisions': total_decisions,
            'decisions_evaluated': len(decisions_with_outcome),
            'success_rate': success_rate,
            'average_confidence': float(avg_confidence),
            'decision_weights': self.decision_weights
        }
