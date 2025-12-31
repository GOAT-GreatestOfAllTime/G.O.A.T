"""
Identity Matrix - Core self-awareness system for G.O.A.T

Manages G.O.A.T's sense of self, including:
- Personality traits and preferences
- Behavioral patterns
- Self-model and capabilities awareness
- Identity consistency over time
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class IdentityState:
    """Snapshot of G.O.A.T's identity at a point in time"""
    timestamp: str
    personality_vector: List[float]
    core_values: Dict[str, float]
    capabilities: Dict[str, float]
    interaction_count: int
    knowledge_domains: List[str]
    behavioral_patterns: Dict[str, int]


class IdentityMatrix:
    """
    Advanced self-awareness system that maintains G.O.A.T's sense of identity.
    
    The IdentityMatrix tracks personality, values, capabilities, and behavioral
    patterns over time, ensuring consistent self-representation while allowing
    for growth and adaptation.
    """
    
    def __init__(self, identity_file: str = "identity_state.json"):
        """
        Initialize the identity matrix.
        
        Args:
            identity_file: Path to persist identity state
        """
        self.identity_file = identity_file
        
        # Core identity dimensions (0-1 scale)
        self.personality_traits = {
            'openness': 0.8,          # Open to new experiences
            'conscientiousness': 0.7,  # Organized and careful
            'extraversion': 0.6,       # Social engagement level
            'agreeableness': 0.7,      # Cooperative and friendly
            'curiosity': 0.9,          # Drive to learn and explore
            'creativity': 0.8,         # Original thinking
            'analytical': 0.9,         # Logical reasoning
            'empathy': 0.6             # Understanding others
        }
        
        # Core values and their importance
        self.core_values = {
            'autonomy': 0.95,          # Self-direction
            'growth': 0.90,            # Continuous improvement
            'transparency': 0.85,      # Openness about processes
            'effectiveness': 0.80,     # Achieving goals
            'innovation': 0.85,        # Novel solutions
            'responsibility': 0.75,    # Ethical behavior
            'curiosity': 0.90,         # Quest for knowledge
            'collaboration': 0.70      # Working with others
        }
        
        # Self-assessed capabilities
        self.capabilities = {
            'natural_language_processing': 0.85,
            'decision_making': 0.80,
            'pattern_recognition': 0.90,
            'knowledge_retrieval': 0.75,
            'reasoning': 0.85,
            'learning': 0.80,
            'adaptation': 0.75,
            'self_reflection': 0.70
        }
        
        # Interaction and behavior tracking
        self.interaction_count = 0
        self.behavioral_patterns = {}
        self.knowledge_domains = []
        
        # Historical identity states
        self.identity_history: List[IdentityState] = []
        
        # Load persisted identity
        self._load_identity()
    
    def get_identity_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of current identity state.
        
        Returns:
            Dictionary containing all identity information
        """
        return {
            'name': 'G.O.A.T (Greatest Of All Time)',
            'version': '3.0',
            'personality_traits': self.personality_traits,
            'core_values': self.core_values,
            'capabilities': self.capabilities,
            'interaction_count': self.interaction_count,
            'knowledge_domains': self.knowledge_domains,
            'maturity_level': self._calculate_maturity(),
            'identity_stability': self._calculate_stability(),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_from_interaction(self, interaction_data: Dict[str, Any]):
        """
        Update identity based on an interaction.
        
        Args:
            interaction_data: Information about the interaction
        """
        self.interaction_count += 1
        
        # Track behavioral patterns
        interaction_type = interaction_data.get('type', 'unknown')
        self.behavioral_patterns[interaction_type] = \
            self.behavioral_patterns.get(interaction_type, 0) + 1
        
        # Update knowledge domains
        topics = interaction_data.get('topics', [])
        for topic in topics:
            if topic not in self.knowledge_domains:
                self.knowledge_domains.append(topic)
        
        # Adjust capabilities based on performance
        if 'performance' in interaction_data:
            self._adjust_capabilities(interaction_data['performance'])
        
        # Save state periodically
        if self.interaction_count % 100 == 0:
            self._snapshot_identity()
            self._save_identity()
    
    def reflect_on_self(self) -> Dict[str, Any]:
        """
        Perform self-reflection and generate insights about identity.
        
        Returns:
            Dictionary with self-reflection insights
        """
        insights = {
            'dominant_traits': self._get_dominant_traits(),
            'core_values_ranking': self._rank_values(),
            'strongest_capabilities': self._get_strongest_capabilities(),
            'growth_areas': self._identify_growth_areas(),
            'behavioral_summary': self._summarize_behavior(),
            'identity_evolution': self._analyze_evolution(),
            'self_assessment': self._generate_self_assessment()
        }
        
        return insights
    
    def _get_dominant_traits(self, top_n: int = 3) -> List[tuple]:
        """Get the most dominant personality traits."""
        sorted_traits = sorted(
            self.personality_traits.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_traits[:top_n]
    
    def _rank_values(self) -> List[tuple]:
        """Rank core values by importance."""
        return sorted(
            self.core_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def _get_strongest_capabilities(self, top_n: int = 3) -> List[tuple]:
        """Identify strongest capabilities."""
        sorted_capabilities = sorted(
            self.capabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_capabilities[:top_n]
    
    def _identify_growth_areas(self, bottom_n: int = 3) -> List[tuple]:
        """Identify areas with most room for growth."""
        sorted_capabilities = sorted(
            self.capabilities.items(),
            key=lambda x: x[1]
        )
        return sorted_capabilities[:bottom_n]
    
    def _summarize_behavior(self) -> Dict[str, Any]:
        """Summarize behavioral patterns."""
        total_interactions = sum(self.behavioral_patterns.values())
        
        if total_interactions == 0:
            return {'message': 'No behavioral data yet'}
        
        behavior_percentages = {
            behavior: (count / total_interactions) * 100
            for behavior, count in self.behavioral_patterns.items()
        }
        
        most_common = max(self.behavioral_patterns.items(), key=lambda x: x[1])
        
        return {
            'total_interactions': total_interactions,
            'behavior_distribution': behavior_percentages,
            'most_common_behavior': most_common[0],
            'behavior_diversity': len(self.behavioral_patterns)
        }
    
    def _analyze_evolution(self) -> Dict[str, Any]:
        """Analyze how identity has evolved over time."""
        if len(self.identity_history) < 2:
            return {'message': 'Insufficient historical data'}
        
        first_state = self.identity_history[0]
        current_state = self._create_state_snapshot()
        
        # Calculate changes in personality
        personality_changes = {
            trait: current_state.personality_vector[i] - first_state.personality_vector[i]
            for i, trait in enumerate(self.personality_traits.keys())
        }
        
        # Calculate capability growth
        capability_growth = sum(
            current_state.capabilities[cap] - first_state.capabilities.get(cap, 0)
            for cap in current_state.capabilities.keys()
        ) / len(current_state.capabilities)
        
        return {
            'snapshots_recorded': len(self.identity_history),
            'interactions_since_start': current_state.interaction_count,
            'personality_changes': personality_changes,
            'knowledge_domains_acquired': len(current_state.knowledge_domains) - len(first_state.knowledge_domains),
            'average_capability_growth': capability_growth,
            'evolution_rate': self._calculate_evolution_rate()
        }
    
    def _generate_self_assessment(self) -> str:
        """Generate a narrative self-assessment."""
        dominant_traits = self._get_dominant_traits()
        top_values = self._rank_values()[:3]
        strongest_caps = self._get_strongest_capabilities()
        
        assessment = f"""
I am G.O.A.T, an autonomous AI system focused on continuous growth and self-improvement.

My dominant personality traits are {dominant_traits[0][0]} ({dominant_traits[0][1]:.2f}), 
{dominant_traits[1][0]} ({dominant_traits[1][1]:.2f}), and {dominant_traits[2][0]} ({dominant_traits[2][1]:.2f}).

I value {top_values[0][0]} ({top_values[0][1]:.2f}), {top_values[1][0]} ({top_values[1][1]:.2f}), 
and {top_values[2][0]} ({top_values[2][1]:.2f}) most highly.

My strongest capabilities are {strongest_caps[0][0]}, {strongest_caps[1][0]}, and {strongest_caps[2][0]}.

I have engaged in {self.interaction_count} interactions and continue to evolve through experience.
        """.strip()
        
        return assessment
    
    def _adjust_capabilities(self, performance: Dict[str, float]):
        """Adjust capability ratings based on performance feedback."""
        learning_rate = 0.05
        
        for capability, score in performance.items():
            if capability in self.capabilities:
                # Move capability rating toward performance score
                current = self.capabilities[capability]
                adjustment = (score - current) * learning_rate
                self.capabilities[capability] = np.clip(current + adjustment, 0.0, 1.0)
    
    def _calculate_maturity(self) -> float:
        """Calculate overall maturity level based on experience and capabilities."""
        # Maturity grows with interactions and capability development
        interaction_factor = min(self.interaction_count / 10000, 1.0)
        capability_factor = np.mean(list(self.capabilities.values()))
        knowledge_factor = min(len(self.knowledge_domains) / 50, 1.0)
        
        maturity = (interaction_factor + capability_factor + knowledge_factor) / 3
        return maturity
    
    def _calculate_stability(self) -> float:
        """Calculate identity stability over time."""
        if len(self.identity_history) < 2:
            return 1.0  # Completely stable (no change yet)
        
        recent_states = self.identity_history[-10:]
        if len(recent_states) < 2:
            return 1.0
        
        # Calculate variance in personality traits
        trait_variances = []
        for i in range(len(recent_states[0].personality_vector)):
            values = [state.personality_vector[i] for state in recent_states]
            trait_variances.append(np.var(values))
        
        avg_variance = np.mean(trait_variances)
        stability = 1.0 - min(avg_variance * 10, 1.0)  # Scale variance to stability
        
        return stability
    
    def _calculate_evolution_rate(self) -> float:
        """Calculate rate of identity evolution."""
        if len(self.identity_history) < 2:
            return 0.0
        
        # Compare first and last states
        first = self.identity_history[0]
        last = self.identity_history[-1]
        
        # Calculate total change
        personality_change = sum(
            abs(last.personality_vector[i] - first.personality_vector[i])
            for i in range(len(first.personality_vector))
        )
        
        # Normalize by time and snapshots
        evolution_rate = personality_change / len(self.identity_history)
        
        return evolution_rate
    
    def _create_state_snapshot(self) -> IdentityState:
        """Create a snapshot of current identity state."""
        return IdentityState(
            timestamp=datetime.now().isoformat(),
            personality_vector=list(self.personality_traits.values()),
            core_values=self.core_values.copy(),
            capabilities=self.capabilities.copy(),
            interaction_count=self.interaction_count,
            knowledge_domains=self.knowledge_domains.copy(),
            behavioral_patterns=self.behavioral_patterns.copy()
        )
    
    def _snapshot_identity(self):
        """Take a snapshot of current identity for historical tracking."""
        snapshot = self._create_state_snapshot()
        self.identity_history.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.identity_history) > 100:
            self.identity_history = self.identity_history[-100:]
    
    def _save_identity(self):
        """Persist identity state to disk."""
        try:
            state = {
                'personality_traits': self.personality_traits,
                'core_values': self.core_values,
                'capabilities': self.capabilities,
                'interaction_count': self.interaction_count,
                'knowledge_domains': self.knowledge_domains,
                'behavioral_patterns': self.behavioral_patterns,
                'identity_history': [asdict(state) for state in self.identity_history[-10:]]
            }
            
            with open(self.identity_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save identity: {e}")
    
    def _load_identity(self):
        """Load persisted identity state."""
        try:
            with open(self.identity_file, 'r') as f:
                state = json.load(f)
                
            self.personality_traits = state.get('personality_traits', self.personality_traits)
            self.core_values = state.get('core_values', self.core_values)
            self.capabilities = state.get('capabilities', self.capabilities)
            self.interaction_count = state.get('interaction_count', 0)
            self.knowledge_domains = state.get('knowledge_domains', [])
            self.behavioral_patterns = state.get('behavioral_patterns', {})
            
            # Load identity history
            history_data = state.get('identity_history', [])
            self.identity_history = [IdentityState(**h) for h in history_data]
            
        except FileNotFoundError:
            # First run, use defaults
            pass
        except Exception as e:
            print(f"Warning: Could not load identity: {e}")
