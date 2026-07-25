"""Pytest configuration and fixtures"""

import pytest


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "filename": "deployment.yaml",
            "filepath": "data/kubernetes/deployment.yaml",
            "content": "apiVersion: apps/v1 kind: Deployment metadata: name: nginx replicas: 3"
        },
        {
            "filename": "service.yaml",
            "filepath": "data/kubernetes/service.yaml",
            "content": "apiVersion: v1 kind: Service metadata: name: nginx ports: 80"
        },
        {
            "filename": "pod.yaml",
            "filepath": "data/kubernetes/pod.yaml",
            "content": "apiVersion: v1 kind: Pod metadata: name: nginx containers: nginx image: nginx"
        },
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        {"id": "q1", "text": "What is a Deployment?"},
        {"id": "q2", "text": "How do I create a Service?"},
        {"id": "q3", "text": "What is a Pod?"},
    ]


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth mappings"""
    return {
        "q1": ["deployment.yaml"],
        "q2": ["service.yaml"],
        "q3": ["pod.yaml"],
    }
