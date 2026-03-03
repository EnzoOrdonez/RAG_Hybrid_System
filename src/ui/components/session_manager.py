"""
Evaluation session manager for within-subjects user study.

Handles: participant login, counterbalanced system ordering,
query assignment, rating collection, SUS questionnaire, and data export.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "data" / "evaluation" / "user_sessions"

SYSTEMS = ["lexical", "semantic", "hybrid"]
SYSTEM_DISPLAY_NAMES = {
    "lexical": "RAG Lexico (BM25)",
    "semantic": "RAG Semantico (Dense)",
    "hybrid": "RAG Hibrido Propuesto",
}

# Counterbalancing: 3 orders (Latin square)
COUNTERBALANCE_ORDERS = [
    ["lexical", "semantic", "hybrid"],
    ["semantic", "hybrid", "lexical"],
    ["hybrid", "lexical", "semantic"],
]

# SUS questions (Spanish)
SUS_QUESTIONS = [
    "Creo que me gustaria usar este sistema frecuentemente",
    "Encontre el sistema innecesariamente complejo",
    "Pense que el sistema era facil de usar",
    "Creo que necesitaria el apoyo de un tecnico para poder usar este sistema",
    "Encontre que las diversas funciones del sistema estaban bien integradas",
    "Pense que habia demasiada inconsistencia en el sistema",
    "Imagino que la mayoria de personas aprenderian a usar este sistema rapidamente",
    "Encontre el sistema muy incomodo de usar",
    "Me senti muy confiado(a) usando el sistema",
    "Necesite aprender muchas cosas antes de poder usar el sistema",
]

# Training queries (not recorded)
TRAINING_QUERIES = [
    "What is cloud computing?",
    "How does a load balancer work?",
    "What is a container?",
]


def _get_evaluation_queries() -> List[dict]:
    """Load 30 queries for evaluation from test_queries.json."""
    queries_path = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
    if not queries_path.exists():
        # Generate a default set
        return _generate_fallback_queries()

    data = json.loads(queries_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("queries", [])

    if not data:
        return _generate_fallback_queries()

    # Select 30 queries: 10 easy, 10 medium, 10 hard
    easy = [q for q in data if q.get("difficulty") == "easy"][:10]
    medium = [q for q in data if q.get("difficulty") == "medium"][:10]
    hard = [q for q in data if q.get("difficulty") == "hard"][:10]

    selected = easy + medium + hard

    # Pad if we don't have enough per category
    if len(selected) < 30:
        remaining = [q for q in data if q not in selected]
        selected.extend(remaining[:30 - len(selected)])

    return selected[:30]


def _generate_fallback_queries() -> List[dict]:
    """Generate fallback queries if test_queries.json not available."""
    queries = []
    templates = [
        ("What is AWS S3 and what are its main features?", "factual", "easy"),
        ("How to set up AWS Lambda for a production workload?", "procedural", "medium"),
        ("What is Azure Virtual Machines and what are its main capabilities?", "factual", "easy"),
        ("How to deploy an application using Azure AKS?", "procedural", "medium"),
        ("What is Google Cloud GKE and what problems does it solve?", "factual", "easy"),
        ("How to deploy a workload on Google Cloud Cloud Run?", "procedural", "medium"),
        ("What are Kubernetes Pods and how do they work?", "factual", "easy"),
        ("How to create and manage Kubernetes Deployments?", "procedural", "medium"),
        ("Compare the serverless compute offerings between AWS Lambda and Azure Functions", "comparative", "hard"),
        ("What are the differences between AWS VPC and Azure Virtual Network?", "comparative", "hard"),
        ("How do managed Kubernetes services compare across AWS EKS, Azure AKS, and GKE?", "comparative", "hard"),
        ("Compare object storage services: AWS S3 vs Azure Blob Storage vs Google Cloud Storage", "comparative", "medium"),
        ("What is the equivalent of AWS IAM in Azure and GCP?", "factual", "easy"),
        ("How does auto-scaling work differently in AWS EC2 vs Azure Virtual Machine Scale Sets?", "comparative", "medium"),
        ("What are the limits and quotas for AWS EC2?", "factual", "easy"),
        ("How to configure auto-scaling in AWS ECS?", "procedural", "medium"),
        ("What are the available SKUs for Azure Blob Storage?", "factual", "easy"),
        ("How to set up monitoring and alerts for Azure Cosmos DB?", "procedural", "medium"),
        ("What machine types are available in Google Cloud Compute Engine?", "factual", "easy"),
        ("How to configure IAM permissions for Google Cloud BigQuery?", "procedural", "medium"),
        ("What is the lifecycle of a Kubernetes StatefulSets?", "factual", "medium"),
        ("How to troubleshoot issues with Kubernetes Services?", "procedural", "medium"),
        ("Compare DNS services: AWS Route 53 vs Azure DNS vs Google Cloud DNS", "comparative", "medium"),
        ("What are the differences between AWS DynamoDB and Azure Cosmos DB?", "comparative", "medium"),
        ("How do load balancing services differ across AWS ELB, Azure Load Balancer, and Google Cloud Load Balancing?", "comparative", "medium"),
        ("Compare the machine learning platforms: AWS SageMaker vs Azure Machine Learning vs Google Vertex AI", "comparative", "hard"),
        ("Compare container orchestration options across AWS ECS, Azure Container Instances, and Google Cloud Run", "comparative", "hard"),
        ("How do encryption at rest implementations differ between AWS KMS, Azure Key Vault, and GCP Cloud KMS?", "comparative", "hard"),
        ("Compare monitoring solutions: AWS CloudWatch vs Azure Monitor vs Google Cloud Monitoring", "comparative", "medium"),
        ("What are the key differences between AWS S3 lifecycle policies and Azure Blob Storage tiering?", "comparative", "medium"),
    ]
    for i, (question, qtype, diff) in enumerate(templates):
        queries.append({
            "query_id": f"eval_{i+1:03d}",
            "question": question,
            "query_type": qtype,
            "difficulty": diff,
        })
    return queries


def get_system_order(participant_id: str) -> List[str]:
    """Deterministic system order based on participant ID.

    Explicit Latin square counterbalancing:
      P01, P04, P07, P10 -> Lexical -> Semantic -> Hybrid
      P02, P05, P08      -> Semantic -> Hybrid -> Lexical
      P03, P06, P09      -> Hybrid -> Lexical -> Semantic
    """
    # Extract numeric part if available (P01 -> 1, P02 -> 2, etc.)
    digits = "".join(c for c in participant_id if c.isdigit())
    if digits:
        num = int(digits)
        idx = (num - 1) % 3  # 0-based: P01->0, P02->1, P03->2, P04->0, ...
    else:
        # Fallback to hash for non-standard IDs
        h = int(hashlib.md5(participant_id.encode()).hexdigest(), 16)
        idx = h % len(COUNTERBALANCE_ORDERS)
    return COUNTERBALANCE_ORDERS[idx]


class EvaluationSession:
    """Manages a single participant's evaluation session."""

    def __init__(self, participant_id: str, experience_level: str):
        self.participant_id = participant_id
        self.experience_level = experience_level
        self.system_order = get_system_order(participant_id)
        self.created_at = datetime.now().isoformat()

        # Load and distribute queries
        all_queries = _get_evaluation_queries()
        self.query_sets = self._distribute_queries(all_queries)

        # Progress tracking
        self.current_system_index = 0
        self.current_query_index = 0
        self.state = "training"  # training, evaluating, break, sus, open_questions, complete

        # Collected data
        self.ratings: List[dict] = []
        self.sus_responses: List[int] = []
        self.open_responses: Dict[str, str] = {}
        self.timestamps: List[dict] = []
        self.training_complete = False

    def _distribute_queries(self, queries: List[dict]) -> Dict[str, List[dict]]:
        """Distribute 30 queries across 3 systems (10 each), rotating by participant."""
        # Split into 3 sets of 10
        sets = [queries[i:i+10] for i in range(0, 30, 10)]

        # Rotate based on participant to counterbalance query-system assignment
        h = int(hashlib.md5(self.participant_id.encode()).hexdigest(), 16)
        rotation = h % 3
        sets = sets[rotation:] + sets[:rotation]

        result = {}
        for i, system_key in enumerate(self.system_order):
            result[system_key] = sets[i] if i < len(sets) else []
        return result

    @property
    def current_system(self) -> Optional[str]:
        if self.current_system_index < len(self.system_order):
            return self.system_order[self.current_system_index]
        return None

    @property
    def current_system_label(self) -> str:
        """Blinded label: Sistema A, B, C."""
        labels = ["Sistema A", "Sistema B", "Sistema C"]
        if self.current_system_index < len(labels):
            return labels[self.current_system_index]
        return "Sistema"

    @property
    def current_queries(self) -> List[dict]:
        sys = self.current_system
        if sys and sys in self.query_sets:
            return self.query_sets[sys]
        return []

    @property
    def current_query(self) -> Optional[dict]:
        queries = self.current_queries
        if self.current_query_index < len(queries):
            return queries[self.current_query_index]
        return None

    @property
    def total_queries_answered(self) -> int:
        return len(self.ratings)

    @property
    def progress_pct(self) -> float:
        total = sum(len(qs) for qs in self.query_sets.values())
        if total == 0:
            return 0.0
        return self.total_queries_answered / total * 100

    def record_rating(
        self,
        utility: int,
        accuracy: int,
        query_shown_ts: float,
        search_clicked_ts: float,
        response_shown_ts: float,
        rated_ts: float,
    ):
        """Record a single query rating."""
        query = self.current_query
        if query is None:
            return

        reading_time_ms = (response_shown_ts - search_clicked_ts) * 1000
        rating_time_ms = (rated_ts - response_shown_ts) * 1000

        rating = {
            "query_id": query.get("query_id", f"q_{self.total_queries_answered}"),
            "question": query.get("question", ""),
            "system": self.current_system,
            "system_label": self.current_system_label,
            "utility_rating": utility,
            "accuracy_rating": accuracy,
            "timestamp_query_shown": query_shown_ts,
            "timestamp_search_clicked": search_clicked_ts,
            "timestamp_response_shown": response_shown_ts,
            "timestamp_rated": rated_ts,
            "reading_time_ms": reading_time_ms,
            "rating_time_ms": rating_time_ms,
        }
        self.ratings.append(rating)

        self.timestamps.append({
            "query_id": rating["query_id"],
            "system": self.current_system,
            "reading_time_ms": reading_time_ms,
            "rating_time_ms": rating_time_ms,
            "total_time_ms": (rated_ts - query_shown_ts) * 1000,
        })

    def advance(self) -> str:
        """Advance to next query or system. Returns new state."""
        self.current_query_index += 1

        if self.current_query_index >= len(self.current_queries):
            # Finished current system
            self.current_system_index += 1
            self.current_query_index = 0

            if self.current_system_index >= len(self.system_order):
                # All systems done
                self.state = "sus"
                return "sus"
            else:
                # Break before next system
                self.state = "break"
                return "break"

        return "evaluating"

    def calculate_sus_score(self) -> float:
        """Calculate SUS score (0-100) from 10 responses."""
        if len(self.sus_responses) != 10:
            return 0.0

        total = 0
        for i, resp in enumerate(self.sus_responses):
            if (i + 1) % 2 == 1:  # Odd items (1,3,5,7,9)
                total += resp - 1
            else:  # Even items (2,4,6,8,10)
                total += 5 - resp

        return total * 2.5

    def get_session_dir(self) -> Path:
        """Get or create session directory."""
        session_dir = SESSIONS_DIR / self.participant_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_checkpoint(self):
        """Save current session state to disk."""
        session_dir = self.get_session_dir()
        data = {
            "participant_id": self.participant_id,
            "experience_level": self.experience_level,
            "system_order": self.system_order,
            "created_at": self.created_at,
            "state": self.state,
            "current_system_index": self.current_system_index,
            "current_query_index": self.current_query_index,
            "training_complete": self.training_complete,
            "ratings": self.ratings,
            "sus_responses": self.sus_responses,
            "open_responses": self.open_responses,
            "timestamps": self.timestamps,
            "query_sets": {k: v for k, v in self.query_sets.items()},
            "checkpoint_time": datetime.now().isoformat(),
        }
        path = session_dir / "session_checkpoint.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, participant_id: str) -> Optional["EvaluationSession"]:
        """Load a saved session checkpoint."""
        path = SESSIONS_DIR / participant_id / "session_checkpoint.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session = cls.__new__(cls)
            session.participant_id = data["participant_id"]
            session.experience_level = data["experience_level"]
            session.system_order = data["system_order"]
            session.created_at = data["created_at"]
            session.state = data["state"]
            session.current_system_index = data["current_system_index"]
            session.current_query_index = data["current_query_index"]
            session.training_complete = data.get("training_complete", False)
            session.ratings = data.get("ratings", [])
            session.sus_responses = data.get("sus_responses", [])
            session.open_responses = data.get("open_responses", {})
            session.timestamps = data.get("timestamps", [])
            session.query_sets = data.get("query_sets", {})
            return session
        except Exception:
            return None

    def export_results(self):
        """Export all session data to individual files."""
        session_dir = self.get_session_dir()

        # session_config.json
        config_data = {
            "participant_id": self.participant_id,
            "experience_level": self.experience_level,
            "system_order": self.system_order,
            "system_display_names": {k: SYSTEM_DISPLAY_NAMES[k] for k in self.system_order},
            "created_at": self.created_at,
            "completed_at": datetime.now().isoformat(),
            "total_queries": self.total_queries_answered,
        }
        (session_dir / "session_config.json").write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ratings.csv
        if self.ratings:
            import csv
            headers = list(self.ratings[0].keys())
            with open(session_dir / "ratings.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.ratings)

        # sus_responses.csv
        if self.sus_responses:
            import csv
            with open(session_dir / "sus_responses.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["question_number", "response"])
                for i, resp in enumerate(self.sus_responses):
                    writer.writerow([i + 1, resp])

        # sus_score.json
        sus_score = self.calculate_sus_score()
        (session_dir / "sus_score.json").write_text(
            json.dumps({
                "raw_scores": self.sus_responses,
                "total": sus_score,
            }, indent=2),
            encoding="utf-8",
        )

        # open_questions.json
        (session_dir / "open_questions.json").write_text(
            json.dumps(self.open_responses, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # timing_log.csv
        if self.timestamps:
            import csv
            headers = list(self.timestamps[0].keys())
            with open(session_dir / "timing_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.timestamps)

        # full_session.json (complete backup)
        full = {
            "config": config_data,
            "ratings": self.ratings,
            "sus_responses": self.sus_responses,
            "sus_score": sus_score,
            "open_responses": self.open_responses,
            "timestamps": self.timestamps,
            "query_sets": self.query_sets,
        }
        (session_dir / "full_session.json").write_text(
            json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return session_dir
