"""
Autonomous Email Agent
Autonomous decision-making and action execution for email intelligence
Author: Sneha Vasudev
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
from config import (
    THREAT_LABELS, SENTIMENT_LABELS, INTENT_LABELS, URGENCY_LABELS,
    OUTPUT_DIR, AUTO_QUARANTINE_ENABLED, AUTO_PRIORITY_ROUTING_ENABLED,
    CRM_LOGGING_ENABLED, SUMMARY_FREQUENCY
)


class EmailAgent:
    """Autonomous agent for email processing and action execution"""
    
    def __init__(self):
        """Initialize agent"""
        self.processed_emails = []
        self.crm_log = []
        self.action_log = []
        self.quarantine_folder = OUTPUT_DIR / "quarantine"
        self.priority_folder = OUTPUT_DIR / "priority"
        self.crm_folder = OUTPUT_DIR / "crm"
        
        # Create folders
        for folder in [self.quarantine_folder, self.priority_folder, self.crm_folder]:
            folder.mkdir(parents=True, exist_ok=True)
    
    def process_email(
        self,
        email_id: str,
        email_text: str,
        predictions: Dict,
        probabilities: Dict
    ) -> Dict:
        """
        Process single email and determine actions
        
        Args:
            email_id: Unique email identifier
            email_text: Email content
            predictions: {task_name: predicted_class}
            probabilities: {task_name: confidence_scores}
        
        Returns:
            Processing result with actions
        """
        
        result = {
            'email_id': email_id,
            'timestamp': datetime.now().isoformat(),
            'text_preview': email_text[:200],
            'predictions': predictions,
            'probabilities': probabilities,
            'actions': [],
            'crm_update': None
        }
        
        # Extract predictions
        threat = predictions.get('threat', 0)
        sentiment = predictions.get('sentiment', 1)
        intent = predictions.get('intent', 0)
        urgency = predictions.get('urgency', 0)
        
        # Threat detection actions
        if AUTO_QUARANTINE_ENABLED and threat > 0:
            action = self.quarantine_email(email_id, email_text, threat)
            result['actions'].append(action)
        
        # Priority routing based on urgency and sentiment
        if AUTO_PRIORITY_ROUTING_ENABLED:
            if urgency == 1 or sentiment == 0:  # High urgency or negative sentiment
                action = self.route_to_priority(email_id)
                result['actions'].append(action)
        
        # CRM logging
        if CRM_LOGGING_ENABLED:
            crm_update = self.log_to_crm(email_id, predictions, sentiment, intent)
            result['crm_update'] = crm_update
        
        self.processed_emails.append(result)
        return result
    
    def quarantine_email(
        self,
        email_id: str,
        email_text: str,
        threat_level: int
    ) -> Dict:
        """
        Quarantine dangerous email
        
        Args:
            email_id: Email identifier
            email_text: Email content
            threat_level: 1=spam, 2=phishing
        
        Returns:
            Action result
        """
        
        threat_name = THREAT_LABELS[threat_level] if threat_level < len(THREAT_LABELS) else "unknown"
        
        filepath = self.quarantine_folder / f"{email_id}_{threat_name}.txt"
        with open(filepath, 'w') as f:
            f.write(f"[QUARANTINED - {threat_name.upper()}]\n")
            f.write(f"Email ID: {email_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("="*70 + "\n")
            f.write(email_text)
        
        action = {
            'type': 'quarantine',
            'email_id': email_id,
            'threat_level': threat_name,
            'filepath': str(filepath),
            'timestamp': datetime.now().isoformat()
        }
        
        self.action_log.append(action)
        return action
    
    def route_to_priority(self, email_id: str) -> Dict:
        """
        Route high-priority email to attention queue
        
        Args:
            email_id: Email identifier
        
        Returns:
            Action result
        """
        
        action = {
            'type': 'priority_route',
            'email_id': email_id,
            'destination': 'high_priority_queue',
            'alert_sent': True,
            'timestamp': datetime.now().isoformat()
        }
        
        self.action_log.append(action)
        return action
    
    def log_to_crm(
        self,
        email_id: str,
        predictions: Dict,
        sentiment: int,
        intent: int
    ) -> Dict:
        """
        Log email intelligence to CRM system
        
        Args:
            email_id: Email identifier
            predictions: All task predictions
            sentiment: Sentiment class
            intent: Intent class
        
        Returns:
            CRM update record
        """
        
        crm_record = {
            'email_id': email_id,
            'timestamp': datetime.now().isoformat(),
            'sentiment': SENTIMENT_LABELS[sentiment] if sentiment < len(SENTIMENT_LABELS) else "unknown",
            'intent': INTENT_LABELS[intent] if intent < len(INTENT_LABELS) else "unknown",
            'threat_level': THREAT_LABELS[predictions.get('threat', 0)],
            'urgency': URGENCY_LABELS[predictions.get('urgency', 0)],
            'followup_required': sentiment == 0 or predictions.get('urgency', 0) == 1  # Negative or urgent
        }
        
        self.crm_log.append(crm_record)
        return crm_record
    
    def batch_process_emails(
        self,
        emails: List[Tuple[str, str, Dict, Dict]]
    ) -> List[Dict]:
        """
        Process batch of emails
        
        Args:
            emails: List of (email_id, text, predictions, probabilities)
        
        Returns:
            List of processing results
        """
        
        results = []
        for email_id, text, preds, probs in emails:
            result = self.process_email(email_id, text, preds, probs)
            results.append(result)
        
        return results
    
    def generate_summary_report(self) -> Dict:
        """
        Generate business intelligence summary
        
        Returns:
            Summary report
        """
        
        if not self.processed_emails:
            return {'message': 'No emails processed'}
        
        df = pd.DataFrame(self.processed_emails)
        
        # Extract threat statistics
        threat_counts = {}
        for pred in df['predictions']:
            threat = pred.get('threat', 0)
            threat_name = THREAT_LABELS[threat] if threat < len(THREAT_LABELS) else "unknown"
            threat_counts[threat_name] = threat_counts.get(threat_name, 0) + 1
        
        # Sentiment statistics
        sentiment_counts = {}
        for pred in df['predictions']:
            sent = pred.get('sentiment', 1)
            sent_name = SENTIMENT_LABELS[sent] if sent < len(SENTIMENT_LABELS) else "unknown"
            sentiment_counts[sent_name] = sentiment_counts.get(sent_name, 0) + 1
        
        # Intent statistics
        intent_counts = {}
        for pred in df['predictions']:
            intent = pred.get('intent', 0)
            intent_name = INTENT_LABELS[intent] if intent < len(INTENT_LABELS) else "unknown"
            intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1
        
        # Action summary
        action_counts = {}
        for action in self.action_log:
            action_type = action.get('type', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'emails_processed': len(self.processed_emails),
            'threats_detected': threat_counts,
            'sentiment_distribution': sentiment_counts,
            'intent_distribution': intent_counts,
            'actions_taken': action_counts,
            'crm_updates': len(self.crm_log),
            'high_priority_emails': sum(1 for a in self.action_log if a['type'] == 'priority_route'),
            'quarantined_emails': sum(1 for a in self.action_log if a['type'] == 'quarantine')
        }
        
        return report
    
    def export_summary_to_file(self) -> Path:
        """Export summary report to file"""
        
        report = self.generate_summary_report()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = OUTPUT_DIR / f"summary_report_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EMAIL INTELLIGENCE SUMMARY REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {report['report_timestamp']}\n")
            f.write(f"Emails Processed: {report['emails_processed']}\n\n")
            
            f.write("THREAT DETECTION:\n")
            f.write("-"*70 + "\n")
            for threat, count in report['threats_detected'].items():
                pct = (count / report['emails_processed'] * 100) if report['emails_processed'] > 0 else 0
                f.write(f"  {threat:<15} {count:>6} ({pct:>5.1f}%)\n")
            
            f.write("\nSENTIMENT ANALYSIS:\n")
            f.write("-"*70 + "\n")
            for sentiment, count in report['sentiment_distribution'].items():
                pct = (count / report['emails_processed'] * 100) if report['emails_processed'] > 0 else 0
                f.write(f"  {sentiment:<15} {count:>6} ({pct:>5.1f}%)\n")
            
            f.write("\nCOMMUNICATION INTENT:\n")
            f.write("-"*70 + "\n")
            for intent, count in report['intent_distribution'].items():
                pct = (count / report['emails_processed'] * 100) if report['emails_processed'] > 0 else 0
                f.write(f"  {intent:<15} {count:>6} ({pct:>5.1f}%)\n")
            
            f.write("\nAUTONOMOUS ACTIONS:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Quarantined: {report['quarantined_emails']}\n")
            f.write(f"  Priority Routed: {report['high_priority_emails']}\n")
            f.write(f"  CRM Updates: {report['crm_updates']}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        return filepath
    
    def get_performance_metrics(self) -> Dict:
        """Calculate agent performance metrics"""
        
        if not self.processed_emails:
            return {}
        
        metrics = {
            'total_emails': len(self.processed_emails),
            'threats_detected': sum(1 for a in self.action_log if a['type'] == 'quarantine'),
            'priority_emails': sum(1 for a in self.action_log if a['type'] == 'priority_route'),
            'crm_updated': len(self.crm_log),
            'detection_rate': (sum(1 for a in self.action_log if a['type'] == 'quarantine') / 
                             len(self.processed_emails) * 100),
        }
        
        return metrics


if __name__ == "__main__":
    print("Email Agent module imported successfully")
