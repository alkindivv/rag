# Monitoring & Analytics Implementation Guide

## 📊 Overview

This document provides a comprehensive implementation guide for monitoring, analytics, and quality assurance systems for the Legal RAG platform. It includes practical code examples, dashboard configurations, and real-time monitoring setups.

**Target Audience**: DevOps Engineers, Backend Developers, Data Engineers  
**Implementation Time**: 2-4 weeks  
**Prerequisites**: Redis, PostgreSQL, Grafana/Prometheus (optional)

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Monitoring      │───▶│   Analytics     │
│   & Response    │    │  Middleware      │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Metrics Store   │    │  Quality        │
                       │  (Redis/DB)      │    │  Assessment     │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Alerting       │    │   Reports       │
                       │   System         │    │   & Insights    │
                       └──────────────────┘    └─────────────────┘
```

---

## 🔧 Core Implementation

### 1. Monitoring Middleware

Create a comprehensive monitoring system that tracks all query interactions:

```python
# src/monitoring/core.py
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from sqlalchemy import create_engine, text
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class QueryType(Enum):
    EXPLICIT_CITATION = "explicit_citation"
    CONTEXTUAL_SEMANTIC = "contextual_semantic"
    DEFINITION = "definition"
    SANCTIONS = "sanctions"
    PROCEDURE = "procedure"
    AUTHORITY = "authority"
    GENERAL = "general"

class ResponseQuality(Enum):
    EXCELLENT = "excellent"  # >0.9
    GOOD = "good"           # 0.7-0.9
    ACCEPTABLE = "acceptable" # 0.5-0.7
    POOR = "poor"           # <0.5

@dataclass
class QueryMetrics:
    query_id: str
    timestamp: datetime
    user_query: str
    query_type: QueryType
    search_strategy: str
    search_duration_ms: float
    llm_duration_ms: float
    total_duration_ms: float
    results_count: int
    reranking_used: bool
    answer_length: int
    citation_count: int
    response_quality: Optional[ResponseQuality] = None
    user_feedback: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    timestamp: datetime
    active_connections: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    db_query_time_ms: float
    embedding_success_rate: float
    llm_availability: bool

class MonitoringService:
    """Core monitoring service for Legal RAG system."""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )
        self.db_engine = create_engine(settings.database_url)
        
    def track_query(self, metrics: QueryMetrics) -> None:
        """Track individual query metrics."""
        try:
            # Store in Redis for real-time access
            redis_key = f"query_metrics:{metrics.query_id}"
            self.redis_client.setex(
                redis_key, 
                86400,  # 24 hours TTL
                json.dumps(asdict(metrics), default=str)
            )
            
            # Store in database for historical analysis
            self._store_query_metrics_db(metrics)
            
            # Update real-time counters
            self._update_realtime_counters(metrics)
            
        except Exception as e:
            logger.error(f"Failed to track query metrics: {e}")
    
    def track_system_health(self, metrics: SystemMetrics) -> None:
        """Track system health metrics."""
        try:
            redis_key = "system_metrics:current"
            self.redis_client.setex(
                redis_key,
                300,  # 5 minutes TTL
                json.dumps(asdict(metrics), default=str)
            )
            
            # Store in time-series for historical analysis
            ts_key = f"system_metrics:ts:{int(time.time())}"
            self.redis_client.setex(ts_key, 3600, json.dumps(asdict(metrics), default=str))
            
        except Exception as e:
            logger.error(f"Failed to track system metrics: {e}")
    
    def get_query_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get query analytics for specified time period."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        query_type,
                        COUNT(*) as total_queries,
                        AVG(total_duration_ms) as avg_duration,
                        AVG(results_count) as avg_results,
                        AVG(answer_length) as avg_answer_length,
                        SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as error_count,
                        AVG(CASE WHEN user_feedback IS NOT NULL THEN user_feedback END) as avg_feedback
                    FROM query_metrics 
                    WHERE timestamp > NOW() - INTERVAL :hours HOUR
                    GROUP BY query_type
                    ORDER BY total_queries DESC
                """)
                
                result = conn.execute(query, {"hours": hours})
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return []
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics summary."""
        try:
            # Get query performance
            query_metrics = self._get_query_performance(hours)
            
            # Get system performance
            system_metrics = self._get_system_performance(hours)
            
            # Get quality metrics
            quality_metrics = self._get_quality_metrics(hours)
            
            return {
                "query_performance": query_metrics,
                "system_performance": system_metrics,
                "quality_metrics": quality_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def _store_query_metrics_db(self, metrics: QueryMetrics) -> None:
        """Store query metrics in database."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO query_metrics (
                        query_id, timestamp, user_query, query_type, search_strategy,
                        search_duration_ms, llm_duration_ms, total_duration_ms,
                        results_count, reranking_used, answer_length, citation_count,
                        response_quality, user_feedback, error_occurred, error_message
                    ) VALUES (
                        :query_id, :timestamp, :user_query, :query_type, :search_strategy,
                        :search_duration_ms, :llm_duration_ms, :total_duration_ms,
                        :results_count, :reranking_used, :answer_length, :citation_count,
                        :response_quality, :user_feedback, :error_occurred, :error_message
                    )
                """)
                
                conn.execute(query, asdict(metrics))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store query metrics in DB: {e}")
    
    def _update_realtime_counters(self, metrics: QueryMetrics) -> None:
        """Update real-time counters for dashboards."""
        current_hour = datetime.now().strftime("%Y%m%d%H")
        
        # Increment counters
        self.redis_client.incr(f"counter:queries:total:{current_hour}")
        self.redis_client.incr(f"counter:queries:{metrics.query_type.value}:{current_hour}")
        
        if metrics.error_occurred:
            self.redis_client.incr(f"counter:errors:{current_hour}")
        
        # Update averages using moving window
        self._update_moving_average("duration_ms", metrics.total_duration_ms)
        self._update_moving_average("results_count", metrics.results_count)
    
    def _update_moving_average(self, metric: str, value: float) -> None:
        """Update moving average for metrics."""
        key = f"moving_avg:{metric}"
        window_size = 100  # Last 100 queries
        
        # Get current values
        values = self.redis_client.lrange(key, 0, -1)
        values = [float(v) for v in values]
        
        # Add new value
        self.redis_client.lpush(key, value)
        
        # Trim to window size
        self.redis_client.ltrim(key, 0, window_size - 1)
        
        # Calculate and store average
        if len(values) < window_size:
            values.append(value)
        else:
            values = values[:-1] + [value]
        
        avg = sum(values) / len(values)
        self.redis_client.setex(f"avg:{metric}", 300, avg)

# Database schema for metrics
METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_metrics (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(36) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    user_query TEXT NOT NULL,
    query_type VARCHAR(50) NOT NULL,
    search_strategy VARCHAR(50) NOT NULL,
    search_duration_ms FLOAT NOT NULL,
    llm_duration_ms FLOAT NOT NULL,
    total_duration_ms FLOAT NOT NULL,
    results_count INTEGER NOT NULL,
    reranking_used BOOLEAN NOT NULL,
    answer_length INTEGER NOT NULL,
    citation_count INTEGER NOT NULL,
    response_quality VARCHAR(20),
    user_feedback FLOAT,
    error_occurred BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON query_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_metrics_type ON query_metrics(query_type);
CREATE INDEX IF NOT EXISTS idx_query_metrics_quality ON query_metrics(response_quality);

CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    active_connections INTEGER NOT NULL,
    memory_usage_mb FLOAT NOT NULL,
    cpu_usage_percent FLOAT NOT NULL,
    cache_hit_rate FLOAT NOT NULL,
    db_query_time_ms FLOAT NOT NULL,
    embedding_success_rate FLOAT NOT NULL,
    llm_availability BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
"""
```

### 2. Integration Decorator

Create decorators to automatically monitor service methods:

```python
# src/monitoring/decorators.py
import functools
import time
import uuid
from datetime import datetime
from typing import Callable, Any
from src.monitoring.core import MonitoringService, QueryMetrics, QueryType
from src.utils.logging import get_logger

logger = get_logger(__name__)
monitoring_service = MonitoringService()

def monitor_query(query_type: QueryType = QueryType.GENERAL):
    """Decorator to monitor query operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            query_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Extract query from arguments
            user_query = kwargs.get('query', args[1] if len(args) > 1 else 'unknown')
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                total_duration = (time.time() - start_time) * 1000
                
                # Extract metrics from result
                metrics = QueryMetrics(
                    query_id=query_id,
                    timestamp=datetime.now(),
                    user_query=str(user_query)[:500],  # Limit length
                    query_type=query_type,
                    search_strategy=result.get('metadata', {}).get('search_type', 'unknown'),
                    search_duration_ms=result.get('metadata', {}).get('duration_ms', 0),
                    llm_duration_ms=0,  # Will be updated by LLM decorator
                    total_duration_ms=total_duration,
                    results_count=len(result.get('results', [])),
                    reranking_used=result.get('metadata', {}).get('reranking_used', False),
                    answer_length=0,  # Will be updated by LLM decorator
                    citation_count=len([r for r in result.get('results', []) if r.citation_string]),
                    error_occurred=False
                )
                
                # Track metrics
                monitoring_service.track_query(metrics)
                
                return result
                
            except Exception as e:
                # Track error
                error_metrics = QueryMetrics(
                    query_id=query_id,
                    timestamp=datetime.now(),
                    user_query=str(user_query)[:500],
                    query_type=query_type,
                    search_strategy='error',
                    search_duration_ms=0,
                    llm_duration_ms=0,
                    total_duration_ms=(time.time() - start_time) * 1000,
                    results_count=0,
                    reranking_used=False,
                    answer_length=0,
                    citation_count=0,
                    error_occurred=True,
                    error_message=str(e)[:1000]
                )
                
                monitoring_service.track_query(error_metrics)
                raise
                
        return wrapper
    return decorator

def monitor_llm(func: Callable) -> Callable:
    """Decorator to monitor LLM operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Update metrics with LLM info
            llm_duration = (time.time() - start_time) * 1000
            answer_length = len(result.get('answer', ''))
            
            # Store additional LLM metrics
            query_id = kwargs.get('query_id', 'unknown')
            monitoring_service.redis_client.setex(
                f"llm_metrics:{query_id}",
                300,
                f"{llm_duration},{answer_length}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM monitoring error: {e}")
            raise
            
    return wrapper
```

### 3. Quality Assessment System

Implement automated quality scoring for answers:

```python
# src/monitoring/quality.py
import re
from typing import Dict, List, Any
from dataclasses import dataclass
from src.monitoring.core import ResponseQuality
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class QualityMetrics:
    relevance_score: float
    citation_accuracy: float
    completeness_score: float
    clarity_score: float
    overall_score: float
    quality_grade: ResponseQuality

class QualityAssessment:
    """Automated quality assessment for Legal RAG responses."""
    
    def __init__(self):
        self.citation_patterns = [
            r'UU No\. \d+ Tahun \d+',
            r'PP No\. \d+ Tahun \d+',
            r'Pasal \d+',
            r'ayat \(\d+\)',
            r'huruf [a-z]'
        ]
    
    def assess_response_quality(
        self, 
        query: str, 
        answer: str, 
        context: List[Dict[str, Any]]
    ) -> QualityMetrics:
        """Assess overall response quality."""
        try:
            # Calculate individual scores
            relevance = self._score_relevance(query, answer)
            citation_accuracy = self._score_citation_accuracy(answer, context)
            completeness = self._score_completeness(query, answer)
            clarity = self._score_clarity(answer)
            
            # Calculate weighted overall score
            weights = {
                'relevance': 0.3,
                'citation_accuracy': 0.3,
                'completeness': 0.25,
                'clarity': 0.15
            }
            
            overall = (
                relevance * weights['relevance'] +
                citation_accuracy * weights['citation_accuracy'] +
                completeness * weights['completeness'] +
                clarity * weights['clarity']
            )
            
            # Determine quality grade
            if overall >= 0.9:
                grade = ResponseQuality.EXCELLENT
            elif overall >= 0.7:
                grade = ResponseQuality.GOOD
            elif overall >= 0.5:
                grade = ResponseQuality.ACCEPTABLE
            else:
                grade = ResponseQuality.POOR
            
            return QualityMetrics(
                relevance_score=relevance,
                citation_accuracy=citation_accuracy,
                completeness_score=completeness,
                clarity_score=clarity,
                overall_score=overall,
                quality_grade=grade
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, ResponseQuality.POOR)
    
    def _score_relevance(self, query: str, answer: str) -> float:
        """Score relevance of answer to query."""
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        # Remove stop words
        stop_words = {'apa', 'yang', 'dalam', 'dan', 'atau', 'adalah', 'dari', 'untuk'}
        query_terms -= stop_words
        answer_terms -= stop_words
        
        if not query_terms:
            return 0.5
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(answer_terms))
        relevance = overlap / len(query_terms)
        
        # Boost for question-specific responses
        if any(q_word in query.lower() for q_word in ['pasal apa', 'uu apa', 'definisi']):
            if any(a_word in answer.lower() for a_word in ['pasal', 'undang-undang', 'adalah']):
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _score_citation_accuracy(self, answer: str, context: List[Dict[str, Any]]) -> float:
        """Score accuracy of citations in answer."""
        # Find citations in answer
        answer_citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            answer_citations.extend(matches)
        
        if not answer_citations:
            return 0.5  # Neutral score if no citations
        
        # Check if citations exist in context
        context_text = ' '.join([item.get('content', '') for item in context])
        valid_citations = 0
        
        for citation in answer_citations:
            if citation.lower() in context_text.lower():
                valid_citations += 1
        
        accuracy = valid_citations / len(answer_citations) if answer_citations else 0
        return accuracy
    
    def _score_completeness(self, query: str, answer: str) -> float:
        """Score completeness of answer."""
        # Basic completeness indicators
        answer_length = len(answer.split())
        
        # Length-based scoring
        if answer_length < 10:
            length_score = 0.3
        elif answer_length < 30:
            length_score = 0.6
        elif answer_length < 100:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Check for key answer components
        components_score = 0.0
        
        # Has direct answer
        if any(word in answer.lower() for word in ['adalah', 'merupakan', 'diatur dalam']):
            components_score += 0.3
        
        # Has citations
        if any(pattern in answer for pattern in ['UU', 'PP', 'Pasal']):
            components_score += 0.3
        
        # Has explanation
        if len(answer.split('.')) > 1:
            components_score += 0.2
        
        # Has structure
        if any(marker in answer for marker in [':', '-', '•', '1.', '2.']):
            components_score += 0.2
        
        return (length_score + components_score) / 2
    
    def _score_clarity(self, answer: str) -> float:
        """Score clarity and readability of answer."""
        # Basic readability metrics
        sentences = answer.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Prefer moderate sentence length
        if 10 <= avg_sentence_length <= 25:
            length_clarity = 1.0
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            length_clarity = 0.8
        else:
            length_clarity = 0.6
        
        # Check for clear structure
        structure_clarity = 0.0
        
        # Has clear formatting
        if any(marker in answer for marker in [':', '\n', '•', '-']):
            structure_clarity += 0.3
        
        # Not too repetitive
        words = answer.lower().split()
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words) if words else 0
        
        if repetition_ratio > 0.7:
            structure_clarity += 0.4
        elif repetition_ratio > 0.5:
            structure_clarity += 0.2
        
        # Uses appropriate legal language
        legal_terms = ['pasal', 'ayat', 'undang-undang', 'peraturan', 'ketentuan']
        if any(term in answer.lower() for term in legal_terms):
            structure_clarity += 0.3
        
        return (length_clarity + structure_clarity) / 2
```

### 4. Dashboard Implementation

Create a real-time dashboard using Flask:

```python
# src/monitoring/dashboard.py
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import json
from src.monitoring.core import MonitoringService
from src.monitoring.quality import QualityAssessment

app = Flask(__name__)
monitoring_service = MonitoringService()
quality_assessment = QualityAssessment()

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/metrics/overview')
def metrics_overview():
    """Get overview metrics for dashboard."""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        # Get performance metrics
        performance = monitoring_service.get_performance_metrics(hours)
        
        # Get query analytics
        analytics = monitoring_service.get_query_analytics(hours)
        
        # Calculate summary stats
        total_queries = sum(item['total_queries'] for item in analytics)
        avg_duration = sum(item['avg_duration'] for item in analytics) / len(analytics) if analytics else 0
        error_rate = sum(item['error_count'] for item in analytics) / total_queries if total_queries > 0 else 0
        
        return jsonify({
            'summary': {
                'total_queries': total_queries,
                'avg_duration_ms': round(avg_duration, 2),
                'error_rate': round(error_rate * 100, 2),
                'system_health': 'healthy' if error_rate < 0.05 else 'warning'
            },
            'performance': performance,
            'analytics': analytics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/realtime')
def realtime_metrics():
    """Get real-time metrics."""
    try:
        current_hour = datetime.now().strftime("%Y%m%d%H")
        
        # Get current counters
        total_queries = monitoring_service.redis_client.get(f"counter:queries:total:{current_hour}") or 0
        errors = monitoring_service.redis_client.get(f"counter:errors:{current_hour}") or 0
        
        # Get current averages
        avg_duration = monitoring_service.redis_client.get("avg:duration_ms") or 0
        avg_results = monitoring_service.redis_client.get("avg:results_count") or 0
        
        # Get system metrics
        system_metrics = monitoring_service.redis_client.get("system_metrics:current")
        system_data = json.loads(system_metrics) if system_metrics else {}
        
        return jsonify({
            'queries_this_hour': int(total_queries),
            'errors_this_hour': int(errors),
            'avg_duration_ms': float(avg_duration),
            'avg_results_count': float(avg_results),
            'system': system_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/quality')
def quality_metrics():
    """Get quality metrics."""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        # Get quality distribution from database
        with monitoring_service.db_engine.connect() as conn:
            from sqlalchemy import text
            
            query = text("""
                SELECT 
                    response_quality,
                    COUNT(*) as count,
                    AVG(user_feedback) as avg_feedback
                FROM query_metrics 
                WHERE timestamp > NOW() - INTERVAL :hours HOUR
                AND response_quality IS NOT NULL
                GROUP BY response_quality
                ORDER BY count DESC
            """)
            
            result = conn.execute(query, {"hours": hours})
            quality_dist = [dict(row._mapping) for row in result]
        
        return jsonify({
            'quality_distribution': quality_dist,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/queries/recent')
def recent_queries():
    """Get recent queries for monitoring."""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        with monitoring_service.db_engine.connect() as conn:
            from sqlalchemy import text
            
            query = text("""
                SELECT 
                    query_id,
                    timestamp,
                    user_query,
                    query_type,
                    total_duration_ms,
                    results_count,
                    response_quality,
                    error_occurred,
                    error_message
                FROM query_metrics 
                ORDER BY timestamp DESC
                LIMIT :limit
            """)
            
            result = conn.execute(query, {"limit": limit})
            queries = [dict(row._mapping) for row in result]
        
        return jsonify({
            'queries': queries,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### 5. Dashboard HTML Template

```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Legal RAG Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .chart-container { height: 300px; position: relative; }
        .recent-queries { max-height: 400px; overflow-y: auto; }
        .query-item { padding: 10px; border-bottom: 1px solid #ecf0f1; }
        .query-error { background-color: #ffeaea; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ Legal RAG Monitoring Dashboard</h1>
        <p>Real-time monitoring and analytics for Indonesian Legal Document RAG System</p>
    </div>

    <div class="metrics-grid">
        <!-- Summary Metrics -->
        <div class="metric-card">
            <div class="metric-value" id="total-queries">-</div>
            <div class="metric-label">Total Queries (24h)</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value" id="avg-duration">-</div>
            
