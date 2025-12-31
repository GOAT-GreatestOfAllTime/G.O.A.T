"""
Semantic Analyzer - Natural language understanding for G.O.A.T

Processes natural language input to extract:
- Meaning and intent
- Entities and relationships
- Sentiment and emotion
- Context and nuance
"""

from typing import List, Dict, Any, Tuple, Optional
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Results of semantic analysis"""
    text: str
    intent: str
    entities: Dict[str, List[str]]
    sentiment: str
    sentiment_score: float
    keywords: List[str]
    topics: List[str]
    complexity: float
    confidence: float


class SemanticAnalyzer:
    """
    Advanced semantic analysis engine for natural language understanding.
    
    The SemanticAnalyzer processes text to extract meaning, intent, and context,
    enabling G.O.A.T to understand and respond appropriately to natural language input.
    """
    
    def __init__(self):
        """Initialize the semantic analyzer."""
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.stopwords = self._load_stopwords()
    
    def analyze(self, text: str, context: Optional[str] = None) -> AnalysisResult:
        """
        Perform comprehensive semantic analysis on input text.
        
        Args:
            text: Input text to analyze
            context: Optional context for disambiguation
            
        Returns:
            AnalysisResult containing all extracted information
        """
        # Preprocess text
        cleaned_text = self._preprocess(text)
        
        # Extract intent
        intent = self._extract_intent(cleaned_text)
        
        # Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Analyze sentiment
        sentiment, sentiment_score = self._analyze_sentiment(cleaned_text)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_text)
        
        # Identify topics
        topics = self._identify_topics(cleaned_text, keywords)
        
        # Calculate complexity
        complexity = self._calculate_complexity(cleaned_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            intent, entities, sentiment_score, keywords
        )
        
        return AnalysisResult(
            text=text,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            keywords=keywords,
            topics=topics,
            complexity=complexity,
            confidence=confidence
        )
    
    def _preprocess(self, text: str) -> str:
        """Clean and normalize input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        return text
    
    def _extract_intent(self, text: str) -> str:
        """
        Determine the primary intent of the text.
        
        Possible intents:
        - question: User is asking for information
        - command: User is requesting an action
        - statement: User is providing information
        - greeting: User is initiating conversation
        - farewell: User is ending conversation
        """
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        
        # Default intent based on punctuation
        if '?' in text:
            return 'question'
        elif any(word in text for word in ['please', 'would you', 'could you', 'can you']):
            return 'request'
        else:
            return 'statement'
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Entity types:
        - person: Names of people
        - organization: Company/organization names
        - location: Places and locations
        - date: Temporal expressions
        - number: Numeric values
        - technical: Technical terms
        """
        entities = {
            'person': [],
            'organization': [],
            'location': [],
            'date': [],
            'number': [],
            'technical': []
        }
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                entities[entity_type].extend(matches)
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of the text.
        
        Returns:
            Tuple of (sentiment_label, sentiment_score)
            sentiment_label: 'positive', 'negative', or 'neutral'
            sentiment_score: -1.0 (very negative) to 1.0 (very positive)
        """
        words = text.split()
        sentiment_score = 0.0
        word_count = 0
        
        for word in words:
            if word in self.sentiment_lexicon:
                sentiment_score += self.sentiment_lexicon[word]
                word_count += 1
        
        # Calculate average sentiment
        if word_count > 0:
            sentiment_score /= word_count
        
        # Determine label
        if sentiment_score > 0.1:
            sentiment_label = 'positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return sentiment_label, sentiment_score
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract the most important keywords from text.
        """
        words = [
            word for word in text.split()
            if word not in self.stopwords and len(word) > 3
        ]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Get top N keywords
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        return keywords
    
    def _identify_topics(self, text: str, keywords: List[str]) -> List[str]:
        """
        Identify main topics discussed in the text.
        """
        topic_mapping = {
            'technology': ['ai', 'software', 'computer', 'digital', 'algorithm', 'data'],
            'business': ['market', 'profit', 'company', 'revenue', 'customer'],
            'science': ['research', 'study', 'experiment', 'theory', 'hypothesis'],
            'finance': ['money', 'investment', 'trading', 'price', 'value'],
            'health': ['medical', 'health', 'treatment', 'disease', 'patient'],
            'education': ['learn', 'teach', 'student', 'knowledge', 'training']
        }
        
        topics = []
        text_lower = text.lower()
        
        for topic, topic_words in topic_mapping.items():
            if any(word in text_lower or word in keywords for word in topic_words):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def _calculate_complexity(self, text: str) -> float:
        """
        Calculate the linguistic complexity of the text.
        
        Returns value between 0.0 (simple) and 1.0 (complex)
        """
        words = text.split()
        
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence count
        sentences = len(re.split(r'[.!?]+', text))
        
        # Words per sentence
        words_per_sentence = len(words) / max(sentences, 1)
        
        # Normalize factors
        length_factor = min(avg_word_length / 10, 1.0)
        sentence_factor = min(words_per_sentence / 20, 1.0)
        
        complexity = (length_factor + sentence_factor) / 2
        
        return complexity
    
    def _calculate_confidence(
        self,
        intent: str,
        entities: Dict[str, List[str]],
        sentiment_score: float,
        keywords: List[str]
    ) -> float:
        """
        Calculate confidence in the analysis.
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence if intent is clear
        if intent in ['question', 'command', 'greeting', 'farewell']:
            confidence += 0.2
        
        # Increase confidence if entities are found
        entity_count = sum(len(v) for v in entities.values())
        confidence += min(entity_count * 0.05, 0.2)
        
        # Increase confidence if sentiment is strong
        if abs(sentiment_score) > 0.3:
            confidence += 0.1
        
        # Increase confidence if keywords are clear
        if len(keywords) >= 5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for intent detection."""
        return {
            'question': [
                r'\bwhat\b', r'\bwho\b', r'\bwhere\b', r'\bwhen\b',
                r'\bwhy\b', r'\bhow\b', r'\?'
            ],
            'command': [
                r'\bdo\b', r'\bmake\b', r'\bcreate\b', r'\brun\b',
                r'\bexecute\b', r'\bstart\b', r'\bstop\b'
            ],
            'greeting': [
                r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgreetings\b',
                r'\bgood morning\b', r'\bgood afternoon\b'
            ],
            'farewell': [
                r'\bgoodbye\b', r'\bbye\b', r'\bsee you\b', r'\bfarewell\b'
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for entity extraction."""
        return {
            'number': [r'\b\d+\.?\d*\b'],
            'date': [r'\b\d{4}-\d{2}-\d{2}\b', r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'],
            'technical': [
                r'\bapi\b', r'\bdatabase\b', r'\balgorithm\b',
                r'\bneural\b', r'\bnetwork\b', r'\bmodel\b'
            ]
        }
    
    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Load sentiment lexicon with word scores."""
        return {
            # Positive words
            'good': 0.5, 'great': 0.7, 'excellent': 0.9, 'amazing': 0.8,
            'wonderful': 0.8, 'fantastic': 0.9, 'love': 0.7, 'like': 0.4,
            'happy': 0.6, 'best': 0.8, 'beautiful': 0.7, 'perfect': 0.9,
            
            # Negative words
            'bad': -0.5, 'terrible': -0.8, 'awful': -0.8, 'horrible': -0.9,
            'hate': -0.7, 'dislike': -0.4, 'sad': -0.6, 'worst': -0.8,
            'ugly': -0.6, 'poor': -0.5, 'fail': -0.7, 'wrong': -0.5
        }
    
    def _load_stopwords(self) -> set:
        """Load stopwords to filter out."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two texts semantically.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with comparison metrics
        """
        analysis1 = self.analyze(text1)
        analysis2 = self.analyze(text2)
        
        # Keyword overlap
        keywords1 = set(analysis1.keywords)
        keywords2 = set(analysis2.keywords)
        keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
        
        # Intent match
        intent_match = analysis1.intent == analysis2.intent
        
        # Sentiment difference
        sentiment_diff = abs(analysis1.sentiment_score - analysis2.sentiment_score)
        
        return {
            'keyword_similarity': keyword_overlap,
            'intent_match': intent_match,
            'sentiment_difference': sentiment_diff,
            'topic_overlap': len(set(analysis1.topics) & set(analysis2.topics)),
            'overall_similarity': (keyword_overlap + (1 if intent_match else 0) + (1 - sentiment_diff)) / 3
        }
