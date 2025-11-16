#!/usr/bin/env python3
"""
API helper for monitoring analysis.
Called by the Next.js API endpoint to fetch or generate analysis data.
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.monitoring.database import MonitoringDatabase
from toolkit.monitoring.analyzer import TrainingAnalyzer


def analyze_job(job_id: str, db_path: str) -> dict:
    """Fetch existing analysis for a job."""
    try:
        db = MonitoringDatabase(db_path)
        analysis = db.get_latest_analysis(job_id)

        if analysis:
            # Get metrics timeline
            samples = db.get_samples_for_job(job_id)

            return {
                'summary': analysis['summary'],
                'recommendations': analysis['recommendations'],
                'log_errors': analysis['log_errors'],
                'health_score': analysis['health_score'],
                'metrics_timeline': samples,
            }
        else:
            # No analysis yet, return empty
            return {
                'summary': None,
                'recommendations': [],
                'log_errors': [],
                'health_score': 'unknown',
                'metrics_timeline': [],
            }

    except Exception as e:
        return {'error': str(e)}


def generate_analysis(job_id: str, db_path: str, job_name: str = '', output_folder: str = 'output') -> dict:
    """Generate new analysis for a job."""
    try:
        analyzer = TrainingAnalyzer(db_path, job_id, output_folder)
        result = analyzer.analyze(job_name if job_name else None)
        return result

    except Exception as e:
        return {'error': str(e)}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: api_helper.py <action> <job_id> <db_path> [job_name] [output_folder]'}))
        sys.exit(1)

    action = sys.argv[1]
    job_id = sys.argv[2]

    if action == 'analyze':
        if len(sys.argv) < 4:
            print(json.dumps({'error': 'Missing db_path'}))
            sys.exit(1)
        db_path = sys.argv[3]
        result = analyze_job(job_id, db_path)
        print(json.dumps(result))

    elif action == 'generate':
        if len(sys.argv) < 4:
            print(json.dumps({'error': 'Missing db_path'}))
            sys.exit(1)
        db_path = sys.argv[3]
        job_name = sys.argv[4] if len(sys.argv) > 4 else ''
        output_folder = sys.argv[5] if len(sys.argv) > 5 else 'output'
        result = generate_analysis(job_id, db_path, job_name, output_folder)
        print(json.dumps(result))

    else:
        print(json.dumps({'error': f'Unknown action: {action}'}))
        sys.exit(1)


if __name__ == '__main__':
    main()
