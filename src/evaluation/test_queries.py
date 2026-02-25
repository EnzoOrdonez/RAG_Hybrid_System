"""
Test Query Dataset Generator for thesis evaluation.

Generates 200 queries with ground truth:
  - 50 AWS-specific
  - 50 Azure-specific
  - 50 GCP-specific
  - 20 Kubernetes-specific
  - 30 cross-cloud (comparative)

Query types (balanced):
  - factual (~33%): "What is a VPC?"
  - procedural (~33%): "How to configure auto-scaling?"
  - comparative (~33%): "Compare EC2 vs Azure VMs"

Difficulty: easy (40%), medium (40%), hard (20%)

Seed: 42 for reproducibility.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestQuery(BaseModel):
    """A test query with ground truth."""
    query_id: str
    question: str
    answer: str = ""
    relevant_chunk_ids: List[str] = []
    cloud_providers: List[str] = []
    query_type: str = "factual"  # factual, procedural, comparative
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # compute, storage, networking, security, ai_ml


# ============================================================
# Cross-cloud queries (hardcoded - most important for thesis)
# ============================================================

CROSS_CLOUD_QUERIES = [
    {
        "question": "Compare the serverless compute offerings between AWS Lambda and Azure Functions",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "compute",
        "difficulty": "medium",
    },
    {
        "question": "What are the differences between AWS VPC and Azure Virtual Network?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "networking",
        "difficulty": "medium",
    },
    {
        "question": "How do managed Kubernetes services compare across AWS EKS, Azure AKS, and GKE?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "compute",
        "difficulty": "hard",
    },
    {
        "question": "Compare object storage services: AWS S3 vs Azure Blob Storage vs Google Cloud Storage",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "storage",
        "difficulty": "medium",
    },
    {
        "question": "What is the equivalent of AWS IAM in Azure and GCP?",
        "query_type": "factual",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "security",
        "difficulty": "easy",
    },
    {
        "question": "Compare container orchestration options across AWS ECS, Azure Container Instances, and Google Cloud Run",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "compute",
        "difficulty": "hard",
    },
    {
        "question": "What are the differences between AWS Lambda cold starts and Azure Functions cold starts?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "compute",
        "difficulty": "hard",
    },
    {
        "question": "How does auto-scaling work differently in AWS EC2 vs Azure Virtual Machine Scale Sets?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "compute",
        "difficulty": "medium",
    },
    {
        "question": "Compare the pricing models of AWS, Azure, and GCP for compute resources",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "compute",
        "difficulty": "hard",
    },
    {
        "question": "What are the main differences between AWS CloudFormation and Azure Resource Manager templates?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "general",
        "difficulty": "medium",
    },
    {
        "question": "Compare monitoring solutions: AWS CloudWatch vs Azure Monitor vs Google Cloud Monitoring",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "general",
        "difficulty": "medium",
    },
    {
        "question": "How do load balancing services differ across AWS ELB, Azure Load Balancer, and Google Cloud Load Balancing?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "networking",
        "difficulty": "medium",
    },
    {
        "question": "What are the differences between AWS DynamoDB and Azure Cosmos DB?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "storage",
        "difficulty": "medium",
    },
    {
        "question": "Compare the container registry services across AWS ECR, Azure Container Registry, and Google Artifact Registry",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "compute",
        "difficulty": "medium",
    },
    {
        "question": "How do serverless database offerings compare between AWS Aurora Serverless and Azure SQL Serverless?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "storage",
        "difficulty": "hard",
    },
    {
        "question": "What is the CNCF equivalent or standard for service mesh across cloud providers?",
        "query_type": "factual",
        "cloud_providers": ["aws", "azure", "gcp", "cncf"],
        "category": "networking",
        "difficulty": "hard",
    },
    {
        "question": "Compare the machine learning platforms: AWS SageMaker vs Azure Machine Learning vs Google Vertex AI",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "ai_ml",
        "difficulty": "hard",
    },
    {
        "question": "What CNCF projects are used for observability across cloud platforms?",
        "query_type": "factual",
        "cloud_providers": ["cncf", "kubernetes"],
        "category": "general",
        "difficulty": "medium",
    },
    {
        "question": "How do the Kubernetes managed services handle cluster upgrades in EKS vs AKS vs GKE?",
        "query_type": "procedural",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "compute",
        "difficulty": "hard",
    },
    {
        "question": "Compare DNS services: AWS Route 53 vs Azure DNS vs Google Cloud DNS",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "networking",
        "difficulty": "medium",
    },
    {
        "question": "What are the key differences between AWS S3 lifecycle policies and Azure Blob Storage tiering?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure"],
        "category": "storage",
        "difficulty": "medium",
    },
    {
        "question": "Compare identity federation approaches across AWS, Azure, and GCP",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "security",
        "difficulty": "hard",
    },
    {
        "question": "How do encryption at rest implementations differ between AWS KMS, Azure Key Vault, and GCP Cloud KMS?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "security",
        "difficulty": "hard",
    },
    {
        "question": "What is the Kubernetes-native approach to secrets management vs cloud provider solutions?",
        "query_type": "comparative",
        "cloud_providers": ["kubernetes", "aws", "azure", "gcp"],
        "category": "security",
        "difficulty": "hard",
    },
    {
        "question": "Compare the CI/CD pipeline offerings: AWS CodePipeline vs Azure DevOps vs Google Cloud Build",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "general",
        "difficulty": "medium",
    },
    {
        "question": "What are the equivalent networking concepts across AWS, Azure, and GCP for VPCs, subnets, and security groups?",
        "query_type": "factual",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "networking",
        "difficulty": "easy",
    },
    {
        "question": "Compare the event-driven architectures: AWS EventBridge vs Azure Event Grid vs Google Eventarc",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "compute",
        "difficulty": "medium",
    },
    {
        "question": "How does the concept of availability zones differ across AWS, Azure, and GCP?",
        "query_type": "comparative",
        "cloud_providers": ["aws", "azure", "gcp"],
        "category": "general",
        "difficulty": "easy",
    },
    {
        "question": "What CNCF graduated projects are most commonly used in multi-cloud deployments?",
        "query_type": "factual",
        "cloud_providers": ["cncf"],
        "category": "general",
        "difficulty": "medium",
    },
    {
        "question": "Compare the Kubernetes ingress controller options across cloud providers and CNCF projects",
        "query_type": "comparative",
        "cloud_providers": ["kubernetes", "aws", "azure", "gcp", "cncf"],
        "category": "networking",
        "difficulty": "hard",
    },
]


# ============================================================
# Per-provider query templates
# ============================================================

PROVIDER_TEMPLATES = {
    "aws": {
        "services": ["EC2", "Lambda", "ECS", "S3", "VPC", "IAM", "EKS", "DynamoDB", "RDS", "CloudWatch"],
        "factual": [
            "What is AWS {service} and what are its main features?",
            "What are the pricing tiers for AWS {service}?",
            "What are the limits and quotas for AWS {service}?",
            "What instance types are available in AWS {service}?",
            "What is the SLA for AWS {service}?",
        ],
        "procedural": [
            "How to set up AWS {service} for a production workload?",
            "How to configure auto-scaling in AWS {service}?",
            "How to monitor AWS {service} with CloudWatch?",
            "How to secure AWS {service} using IAM policies?",
            "How to deploy a containerized application on AWS {service}?",
        ],
    },
    "azure": {
        "services": ["Virtual Machines", "Functions", "AKS", "Blob Storage", "Virtual Network",
                      "Entra ID", "Cosmos DB", "Container Instances", "App Service", "Monitor"],
        "factual": [
            "What is Azure {service} and what are its main capabilities?",
            "What are the available SKUs for Azure {service}?",
            "What are the service limits for Azure {service}?",
            "What regions support Azure {service}?",
            "What compliance certifications does Azure {service} have?",
        ],
        "procedural": [
            "How to deploy an application using Azure {service}?",
            "How to configure networking for Azure {service}?",
            "How to set up monitoring and alerts for Azure {service}?",
            "How to implement backup and disaster recovery for Azure {service}?",
            "How to scale Azure {service} horizontally?",
        ],
    },
    "gcp": {
        "services": ["Compute Engine", "Cloud Functions", "GKE", "Cloud Storage",
                      "Cloud SQL", "BigQuery", "VPC", "Cloud Run", "Pub/Sub", "Cloud Build"],
        "factual": [
            "What is Google Cloud {service} and what problems does it solve?",
            "What machine types are available in Google Cloud {service}?",
            "What are the quotas for Google Cloud {service}?",
            "What is the pricing model for Google Cloud {service}?",
            "What are the best practices for Google Cloud {service}?",
        ],
        "procedural": [
            "How to deploy a workload on Google Cloud {service}?",
            "How to configure IAM permissions for Google Cloud {service}?",
            "How to set up logging and monitoring for Google Cloud {service}?",
            "How to implement auto-scaling with Google Cloud {service}?",
            "How to migrate an on-premises application to Google Cloud {service}?",
        ],
    },
    "kubernetes": {
        "services": ["Pods", "Deployments", "Services", "Ingress", "ConfigMaps",
                      "Secrets", "StatefulSets", "DaemonSets", "HPA", "RBAC"],
        "factual": [
            "What are Kubernetes {service} and how do they work?",
            "What is the lifecycle of a Kubernetes {service}?",
            "What are the best practices for Kubernetes {service}?",
            "What are the resource limits for Kubernetes {service}?",
        ],
        "procedural": [
            "How to create and manage Kubernetes {service}?",
            "How to troubleshoot issues with Kubernetes {service}?",
            "How to configure Kubernetes {service} for high availability?",
            "How to update Kubernetes {service} without downtime?",
        ],
    },
}

DIFFICULTY_WEIGHTS = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
CATEGORY_MAP = {
    "EC2": "compute", "Lambda": "compute", "ECS": "compute", "EKS": "compute",
    "S3": "storage", "DynamoDB": "storage", "RDS": "storage",
    "VPC": "networking", "IAM": "security", "CloudWatch": "general",
    "Virtual Machines": "compute", "Functions": "compute", "AKS": "compute",
    "Blob Storage": "storage", "Virtual Network": "networking",
    "Entra ID": "security", "Cosmos DB": "storage",
    "Container Instances": "compute", "App Service": "compute", "Monitor": "general",
    "Compute Engine": "compute", "Cloud Functions": "compute", "GKE": "compute",
    "Cloud Storage": "storage", "Cloud SQL": "storage", "BigQuery": "storage",
    "Cloud Run": "compute", "Pub/Sub": "general", "Cloud Build": "general",
    "Pods": "compute", "Deployments": "compute", "Services": "networking",
    "Ingress": "networking", "ConfigMaps": "general", "Secrets": "security",
    "StatefulSets": "compute", "DaemonSets": "compute", "HPA": "compute",
    "RBAC": "security",
}


def _find_relevant_chunks(
    question: str,
    cloud_providers: List[str],
    hybrid_index=None,
    max_chunks: int = 3,
) -> List[str]:
    """Find relevant chunk IDs for a question using retrieval."""
    if hybrid_index is None:
        return []

    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.query_processor import QueryProcessor

        qp = QueryProcessor()
        retriever = HybridRetriever(hybrid_index, query_processor=qp)
        results = retriever.search(question, top_k=max_chunks)
        return [r.chunk_id for r in results]
    except Exception as e:
        logger.debug("Chunk retrieval failed for '%s': %s", question[:50], e)
        return []


def generate_provider_queries(
    provider: str,
    count: int,
    hybrid_index=None,
    seed: int = 42,
) -> List[TestQuery]:
    """Generate queries for a single provider from templates."""
    rng = random.Random(seed + hash(provider))
    config = PROVIDER_TEMPLATES.get(provider, {})
    services = config.get("services", [])
    factual_templates = config.get("factual", [])
    procedural_templates = config.get("procedural", [])

    queries = []
    idx = 0

    while len(queries) < count and idx < count * 3:
        idx += 1
        service = rng.choice(services)

        # Alternate between factual and procedural
        if rng.random() < 0.5:
            query_type = "factual"
            template = rng.choice(factual_templates)
        else:
            query_type = "procedural"
            template = rng.choice(procedural_templates)

        question = template.format(service=service)

        # Assign difficulty
        r = rng.random()
        if r < 0.4:
            difficulty = "easy"
        elif r < 0.8:
            difficulty = "medium"
        else:
            difficulty = "hard"

        category = CATEGORY_MAP.get(service, "general")

        # Find relevant chunks
        relevant_ids = _find_relevant_chunks(
            question, [provider], hybrid_index, max_chunks=3
        )

        queries.append(TestQuery(
            query_id=f"q_{provider}_{len(queries)+1:03d}",
            question=question,
            relevant_chunk_ids=relevant_ids,
            cloud_providers=[provider],
            query_type=query_type,
            difficulty=difficulty,
            category=category,
        ))

    return queries[:count]


def generate_cross_cloud_queries(
    hybrid_index=None,
) -> List[TestQuery]:
    """Generate cross-cloud queries from hardcoded list."""
    queries = []
    for i, q in enumerate(CROSS_CLOUD_QUERIES):
        relevant_ids = _find_relevant_chunks(
            q["question"],
            q["cloud_providers"],
            hybrid_index,
            max_chunks=5,
        )
        queries.append(TestQuery(
            query_id=f"q_cross_{i+1:03d}",
            question=q["question"],
            relevant_chunk_ids=relevant_ids,
            cloud_providers=q["cloud_providers"],
            query_type=q["query_type"],
            difficulty=q["difficulty"],
            category=q["category"],
        ))
    return queries


def generate_all_queries(
    count: int = 200,
    hybrid_index=None,
    seed: int = 42,
) -> List[TestQuery]:
    """Generate the full test query dataset."""
    all_queries = []

    # Distribution: AWS=50, Azure=50, GCP=50, K8s=20, cross-cloud=30
    cross_cloud_count = min(30, len(CROSS_CLOUD_QUERIES))
    remaining = count - cross_cloud_count
    provider_counts = {
        "aws": int(remaining * 50 / 170),
        "azure": int(remaining * 50 / 170),
        "gcp": int(remaining * 50 / 170),
        "kubernetes": remaining - 3 * int(remaining * 50 / 170),
    }

    # Generate per-provider queries
    for provider, cnt in provider_counts.items():
        logger.info("Generating %d queries for %s...", cnt, provider)
        pq = generate_provider_queries(provider, cnt, hybrid_index, seed)
        all_queries.extend(pq)

    # Generate cross-cloud queries
    logger.info("Generating %d cross-cloud queries...", cross_cloud_count)
    cq = generate_cross_cloud_queries(hybrid_index)
    all_queries.extend(cq)

    # Reassign sequential IDs
    for i, q in enumerate(all_queries):
        q.query_id = f"q{i+1:03d}"

    return all_queries


def save_queries(queries: List[TestQuery], output_path: str):
    """Save queries to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [q.model_dump() for q in queries]
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d queries to %s", len(queries), output_path)


def load_queries(path: str) -> List[TestQuery]:
    """Load queries from JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TestQuery(**d) for d in data]


def verify_queries(queries: List[TestQuery]) -> Dict:
    """Verify the query dataset and return stats."""
    stats = {
        "total": len(queries),
        "with_ground_truth": sum(1 for q in queries if q.relevant_chunk_ids),
        "by_provider": {},
        "by_type": {},
        "by_difficulty": {},
        "by_category": {},
        "cross_cloud": sum(1 for q in queries if len(q.cloud_providers) > 1),
    }
    for q in queries:
        for p in q.cloud_providers:
            stats["by_provider"][p] = stats["by_provider"].get(p, 0) + 1
        stats["by_type"][q.query_type] = stats["by_type"].get(q.query_type, 0) + 1
        stats["by_difficulty"][q.difficulty] = stats["by_difficulty"].get(q.difficulty, 0) + 1
        stats["by_category"][q.category] = stats["by_category"].get(q.category, 0) + 1
    return stats
