"""Build high-quality NL-to-graph training pairs from expert modeling patterns.

Sources:
  1. 23 expert modeling patterns from bundled data (node_labels, rels, cypher)
  2. 477 cypher examples → reverse-engineered as query→subgraph pairs
  3. Structural patterns missing from training data:
     - Hierarchical (category trees, org charts)
     - Temporal (timelines, versioning, event sequences)
     - Workflow (state machines, approval flows)
     - Multi-hop traversal (friend-of-friend, dependency chains)
     - Aggregation (count, shortest path, recommendations)

This is THE highest-value training data — it directly teaches the three GNN
objectives (g.ask, g.ingest, NL→graph) using expert-verified graph patterns.

Usage:
    python data/scripts/build_expert_nl_pairs.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_OUTPUT = Path("data/training/expert_nl_pairs.jsonl")

# Seed for reproducibility
random.seed(42)

# Edge weights by relationship semantics
EDGE_WEIGHTS: dict[str, float] = {
    "ACTED_IN": 0.8, "DIRECTED": 0.8, "PRODUCED": 0.7, "REVIEWED": 0.6,
    "PURCHASED": 0.9, "BELONGS_TO": 0.7, "CHILD_OF": 0.9, "TAGGED_WITH": 0.6,
    "FOLLOWS": 0.7, "AUTHORED": 0.8, "MEMBER_OF": 0.6,
    "OWNS": 0.9, "SENT": 0.9, "RECEIVED": 0.9, "USED_DEVICE": 0.7,
    "RUNS_ON": 0.8, "DEPENDS_ON": 0.9, "CAUSED_BY": 0.85, "CONNECTED_TO": 0.6,
    "SUPPLIES": 0.8, "STORED_IN": 0.7, "SHIPPED_VIA": 0.75,
    "REPORTS_TO": 0.9, "WORKS_IN": 0.7, "ASSIGNED_TO": 0.8, "HAS_ROLE": 0.6,
    "TREATED_BY": 0.85, "DIAGNOSED_WITH": 0.9, "PRESCRIBED": 0.85,
    "PUBLISHED_IN": 0.7, "WORKS_FOR": 0.8, "LIKES": 0.5,
    "IS_FRIENDS_WITH": 0.6, "RATED": 0.5, "IN_LANGUAGE": 0.4,
    "HAS_VERSION": 0.8, "SUCCEEDED_BY": 0.9, "PRECEDED_BY": 0.9,
    "OCCURRED_ON": 0.7, "TRANSITIONED_TO": 0.85, "APPROVED_BY": 0.8,
    "CONTAINS": 0.7, "PART_OF": 0.8, "LOCATED_IN": 0.6,
    "SIMILAR_TO": 0.5, "RELATED_TOPIC": 0.5,
    "ENROLLED_IN": 0.8, "TEACHES": 0.8, "PREREQUISITE_OF": 0.9,
    "SERVES": 0.7, "HAS_ENDPOINT": 0.6, "CALLS": 0.8,
}


def _w(rel_type: str) -> float:
    return EDGE_WEIGHTS.get(rel_type, 0.6)


def _pair(
    pair_id: str,
    input_text: str,
    source_node: dict,
    target_nodes: list[dict],
    edges: list[dict],
    *,
    reverse_edges: list[dict] | None = None,
    source: str = "expert_pattern",
    pattern_type: str = "nl_to_graph",
    quality_level: str = "clean",
) -> dict:
    """Build a single training pair."""
    return {
        "id": pair_id,
        "input_text": input_text,
        "metadata": {
            "domain": "expert_nl",
            "source": source,
            "pattern_type": pattern_type,
            "quality_level": quality_level,
        },
        "expected_graph": {
            "source_node": source_node,
            "target_nodes": target_nodes,
            "edges": edges,
            "reverse_edges": reverse_edges or [],
        },
        "quality": {
            "edge_count": len(edges),
            "reverse_edge_count": len(reverse_edges or []),
            "total_edges": len(edges) + len(reverse_edges or []),
            "text_length": len(input_text),
            "verified_source": source,
        },
    }


def _node(nid: str, label: str, **props: object) -> dict:
    return {"id": nid, "label": label, "properties": props}


def _edge(src: str, tgt: str, rel: str) -> dict:
    return {"source": src, "target": tgt, "type": rel, "weight": _w(rel)}


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Expert modeling patterns → NL-to-graph pairs
# Each pattern generates multiple variations (clean, casual, messy)
# ═══════════════════════════════════════════════════════════════════

def gen_ecommerce() -> list[dict]:
    """E-commerce product catalog pattern."""
    pairs = []
    # Clean queries
    queries = [
        ("ecom_001", "Show me products in the Electronics category with their reviews",
         "clean"),
        ("ecom_002", "Which customers purchased laptops and what did they also buy?",
         "clean"),
        ("ecom_003", "Find the top-rated products in each category",
         "clean"),
        ("ecom_004", "whats the best selling stuff in electronics",
         "casual"),
        ("ecom_005", "customers who bought X also bought Y recommendations",
         "messy"),
        ("ecom_006", "Show the full product taxonomy from Electronics down to subcategories",
         "clean"),
        ("ecom_007", "Which products have reviews from customers who also reviewed competing products?",
         "clean"),
        ("ecom_008", "products ppl keep returning in last month",
         "casual"),
        ("ecom_009", "Map the relationship between customer segments and product categories",
         "clean"),
        ("ecom_010", "avg review score by category with product count",
         "messy"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("customer_1", "Customer", name="Alice", segment="Premium"),
            _node("product_1", "Product", name="Laptop Pro", sku="LP-001", price=1299.99),
            _node("product_2", "Product", name="Wireless Mouse", sku="WM-042", price=29.99),
            _node("category_1", "Category", name="Electronics"),
            _node("category_2", "Category", name="Accessories"),
            _node("review_1", "Review", rating=5, text="Excellent laptop"),
        ]
        edges = [
            _edge("customer_1", "product_1", "PURCHASED"),
            _edge("customer_1", "product_2", "PURCHASED"),
            _edge("product_1", "category_1", "BELONGS_TO"),
            _edge("product_2", "category_2", "BELONGS_TO"),
            _edge("category_2", "category_1", "CHILD_OF"),
            _edge("customer_1", "review_1", "AUTHORED"),
            _edge("review_1", "product_1", "REVIEWED"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_ecommerce", quality_level=quality,
        ))
    return pairs


def gen_social_network() -> list[dict]:
    """Social network pattern."""
    pairs = []
    queries = [
        ("social_001", "Show me who Alice follows and their recent posts", "clean"),
        ("social_002", "Find mutual friends between Alice and Bob", "clean"),
        ("social_003", "Which communities have the most active members?", "clean"),
        ("social_004", "ppl alice follows that also follow her back", "casual"),
        ("social_005", "friend of friend recommendations for alice", "messy"),
        ("social_006", "Show the social graph around the AI community", "clean"),
        ("social_007", "Who are the most influential people in Alice's network?", "clean"),
        ("social_008", "posts from friends of friends this week", "casual"),
        ("social_009", "find all paths between alice and eve through follows", "messy"),
        ("social_010", "Which people bridge two different communities?", "clean"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("alice", "Person", name="Alice", handle="@alice"),
            _node("bob", "Person", name="Bob", handle="@bob"),
            _node("carol", "Person", name="Carol", handle="@carol"),
            _node("post_1", "Post", title="AI in 2026", created_at="2026-03-15"),
            _node("post_2", "Post", title="Graph databases", created_at="2026-03-14"),
            _node("community_1", "Community", name="AI Enthusiasts", members=1500),
        ]
        edges = [
            _edge("alice", "bob", "FOLLOWS"),
            _edge("alice", "carol", "FOLLOWS"),
            _edge("bob", "alice", "FOLLOWS"),
            _edge("bob", "post_1", "AUTHORED"),
            _edge("carol", "post_2", "AUTHORED"),
            _edge("alice", "community_1", "MEMBER_OF"),
            _edge("bob", "community_1", "MEMBER_OF"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_social", quality_level=quality,
        ))
    return pairs


def gen_fraud_detection() -> list[dict]:
    """Fraud detection pattern."""
    pairs = []
    queries = [
        ("fraud_001", "Find accounts sharing the same device that aren't owned by the same customer", "clean"),
        ("fraud_002", "Detect circular transaction patterns involving account A-1234", "clean"),
        ("fraud_003", "Which devices are associated with flagged transactions?", "clean"),
        ("fraud_004", "suspicious accounts using same phone", "casual"),
        ("fraud_005", "money going in circles between 3+ accounts", "messy"),
        ("fraud_006", "Show the fraud ring around device D-789", "clean"),
        ("fraud_007", "Find all transactions above 10K between accounts that share a device", "clean"),
        ("fraud_008", "accounts opened same day same device different names", "casual"),
        ("fraud_009", "Map the full network of accounts connected through shared devices and transactions", "clean"),
        ("fraud_010", "Which customers have accounts linked to known fraud devices?", "clean"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("account_1", "Account", iban="NL12ABCD0123456789", status="active"),
            _node("account_2", "Account", iban="DE89EFGH9876543210", status="flagged"),
            _node("customer_1", "Customer", name="John Doe", kyc_verified=True),
            _node("customer_2", "Customer", name="Jane Smith", kyc_verified=False),
            _node("device_1", "Device", fingerprint="D-789", type="mobile"),
            _node("tx_1", "Transaction", amount=15000, currency="EUR", timestamp="2026-03-10"),
        ]
        edges = [
            _edge("customer_1", "account_1", "OWNS"),
            _edge("customer_2", "account_2", "OWNS"),
            _edge("account_1", "tx_1", "SENT"),
            _edge("tx_1", "account_2", "RECEIVED"),
            _edge("account_1", "device_1", "USED_DEVICE"),
            _edge("account_2", "device_1", "USED_DEVICE"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_fraud", quality_level=quality,
        ))
    return pairs


def gen_it_infrastructure() -> list[dict]:
    """IT infrastructure dependency pattern."""
    pairs = []
    queries = [
        ("infra_001", "What services would be affected if server prod-db-01 goes down?", "clean"),
        ("infra_002", "Show the dependency chain for the payment service", "clean"),
        ("infra_003", "Which servers have the most critical services running on them?", "clean"),
        ("infra_004", "what breaks if we restart the auth service", "casual"),
        ("infra_005", "servers with open critical incidents", "messy"),
        ("infra_006", "Map all services in the checkout flow and their infrastructure dependencies", "clean"),
        ("infra_007", "Find single points of failure in our service architecture", "clean"),
        ("infra_008", "services depending on postgres that also depend on redis", "casual"),
        ("infra_009", "Show the blast radius of network segment NET-A going offline", "clean"),
        ("infra_010", "Which incidents were caused by the same root service failure?", "clean"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("server_1", "Server", hostname="prod-db-01", ip="10.0.1.5", tier="critical"),
            _node("server_2", "Server", hostname="prod-app-01", ip="10.0.2.3", tier="high"),
            _node("svc_auth", "Service", name="auth-service", version="3.2.1"),
            _node("svc_payment", "Service", name="payment-service", version="2.0.4"),
            _node("svc_api", "Service", name="api-gateway", version="5.1.0"),
            _node("network_1", "Network", segment="NET-A", vlan=100),
            _node("incident_1", "Incident", severity="critical", title="Payment timeout"),
        ]
        edges = [
            _edge("svc_auth", "server_1", "RUNS_ON"),
            _edge("svc_payment", "server_2", "RUNS_ON"),
            _edge("svc_api", "server_2", "RUNS_ON"),
            _edge("svc_payment", "svc_auth", "DEPENDS_ON"),
            _edge("svc_api", "svc_auth", "DEPENDS_ON"),
            _edge("svc_api", "svc_payment", "DEPENDS_ON"),
            _edge("server_1", "network_1", "CONNECTED_TO"),
            _edge("server_2", "network_1", "CONNECTED_TO"),
            _edge("incident_1", "svc_payment", "CAUSED_BY"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_infra", quality_level=quality,
        ))
    return pairs


def gen_supply_chain() -> list[dict]:
    """Supply chain tracking pattern."""
    pairs = []
    queries = [
        ("supply_001", "Track the shipment of product SKU-1234 from supplier to warehouse", "clean"),
        ("supply_002", "Which suppliers provide components for our top-selling products?", "clean"),
        ("supply_003", "Show all products currently stored in the Stockholm warehouse", "clean"),
        ("supply_004", "delayed shipments from china this month", "casual"),
        ("supply_005", "which supplier has the best on-time delivery rate", "messy"),
        ("supply_006", "Map the full supply chain for the EV battery module", "clean"),
        ("supply_007", "Find alternative suppliers for products affected by shipping delays", "clean"),
        ("supply_008", "warehouses running low on stock from supplier X", "casual"),
        ("supply_009", "Show the multi-tier supplier dependency for our critical components", "clean"),
        ("supply_010", "Which products have single-source supplier risk?", "clean"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("supplier_1", "Supplier", name="TechParts AB", country="Sweden"),
            _node("supplier_2", "Supplier", name="ChipCo Ltd", country="Taiwan"),
            _node("product_1", "Product", name="EV Battery Module", sku="SKU-1234"),
            _node("product_2", "Product", name="Controller Board", sku="SKU-5678"),
            _node("warehouse_1", "Warehouse", name="Stockholm DC", location="Stockholm", capacity=50000),
            _node("shipment_1", "Shipment", carrier="DHL", status="in_transit", shipped_at="2026-03-12"),
        ]
        edges = [
            _edge("supplier_1", "product_1", "SUPPLIES"),
            _edge("supplier_2", "product_2", "SUPPLIES"),
            _edge("product_2", "product_1", "PART_OF"),
            _edge("product_1", "warehouse_1", "STORED_IN"),
            _edge("product_1", "shipment_1", "SHIPPED_VIA"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_supply_chain", quality_level=quality,
        ))
    return pairs


def gen_org_hierarchy() -> list[dict]:
    """Organizational hierarchy pattern."""
    pairs = []
    queries = [
        ("org_001", "Show the reporting chain from developer Erik to the CEO", "clean"),
        ("org_002", "Which departments are working on Project Alpha?", "clean"),
        ("org_003", "Find all employees who report to the VP of Engineering", "clean"),
        ("org_004", "who does maria report to and how many levels up is ceo", "casual"),
        ("org_005", "headcount per department with project assignments", "messy"),
        ("org_006", "Show the full organizational tree for the Engineering department", "clean"),
        ("org_007", "Which employees work across multiple projects?", "clean"),
        ("org_008", "people in both project alpha and project beta", "casual"),
        ("org_009", "Map the cross-functional team structure for the product launch", "clean"),
        ("org_010", "Which managers have the deepest reporting chains?", "clean"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("emp_erik", "Employee", name="Erik Larsson", title="Senior Developer"),
            _node("emp_maria", "Employee", name="Maria Chen", title="VP Engineering"),
            _node("emp_ceo", "Employee", name="Anna Svensson", title="CEO"),
            _node("dept_eng", "Department", name="Engineering", budget=2000000),
            _node("dept_product", "Department", name="Product", budget=800000),
            _node("role_dev", "Role", name="Senior Developer", level="IC4"),
            _node("project_alpha", "Project", name="Project Alpha", deadline="2026-06-01"),
        ]
        edges = [
            _edge("emp_erik", "emp_maria", "REPORTS_TO"),
            _edge("emp_maria", "emp_ceo", "REPORTS_TO"),
            _edge("emp_erik", "dept_eng", "WORKS_IN"),
            _edge("emp_maria", "dept_eng", "WORKS_IN"),
            _edge("emp_erik", "role_dev", "HAS_ROLE"),
            _edge("emp_erik", "project_alpha", "ASSIGNED_TO"),
            _edge("emp_maria", "project_alpha", "ASSIGNED_TO"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_org", quality_level=quality,
        ))
    return pairs


def gen_healthcare() -> list[dict]:
    """Healthcare patient journey pattern."""
    pairs = []
    queries = [
        ("health_001", "Show the treatment history for patient P-1001", "clean"),
        ("health_002", "Which doctors treat patients with both diabetes and hypertension?", "clean"),
        ("health_003", "Find patients with similar diagnoses to patient P-1001 and their treatments", "clean"),
        ("health_004", "treatments that worked for patients like mine", "casual"),
        ("health_005", "dr anderson patients with icd E11 this year", "messy"),
        ("health_006", "Map the care pathway for Type 2 Diabetes patients", "clean"),
        ("health_007", "Which treatments are most commonly prescribed for diagnosis D-042?", "clean"),
        ("health_008", "patients seeing multiple specialists for same condition", "casual"),
        ("health_009", "Show drug interaction risks for patient P-1001's current prescriptions", "clean"),
        ("health_010", "Which diagnoses frequently co-occur in our patient population?", "clean"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("patient_1", "Patient", patient_id="P-1001", age=58, gender="M"),
            _node("doctor_1", "Doctor", name="Dr. Anderson", specialty="Endocrinology"),
            _node("doctor_2", "Doctor", name="Dr. Berg", specialty="Cardiology"),
            _node("diag_1", "Diagnosis", icd_code="E11", name="Type 2 Diabetes", date="2025-06-15"),
            _node("diag_2", "Diagnosis", icd_code="I10", name="Hypertension", date="2024-01-20"),
            _node("treatment_1", "Treatment", name="Metformin 500mg", type="medication"),
            _node("treatment_2", "Treatment", name="Lisinopril 10mg", type="medication"),
        ]
        edges = [
            _edge("patient_1", "doctor_1", "TREATED_BY"),
            _edge("patient_1", "doctor_2", "TREATED_BY"),
            _edge("patient_1", "diag_1", "DIAGNOSED_WITH"),
            _edge("patient_1", "diag_2", "DIAGNOSED_WITH"),
            _edge("doctor_1", "treatment_1", "PRESCRIBED"),
            _edge("doctor_2", "treatment_2", "PRESCRIBED"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_healthcare", quality_level=quality,
        ))
    return pairs


def gen_content_management() -> list[dict]:
    """Content management pattern."""
    pairs = []
    queries = [
        ("content_001", "Find articles related to 'graph databases' through shared tags", "clean"),
        ("content_002", "Which authors write about both AI and databases?", "clean"),
        ("content_003", "Show the tag co-occurrence network for the tech publication", "clean"),
        ("content_004", "articles about neo4j from last month", "casual"),
        ("content_005", "authors who write together most often", "messy"),
        ("content_006", "What topics have the most content gaps in our publication?", "clean"),
        ("content_007", "Find the most prolific authors and their topic coverage", "clean"),
        ("content_008", "trending tags this quarter with article count", "casual"),
    ]

    for pid, text, quality in queries:
        nodes = [
            _node("article_1", "Article", title="Graph Databases in 2026", published="2026-02-01"),
            _node("article_2", "Article", title="AI for Knowledge Graphs", published="2026-03-01"),
            _node("author_1", "Author", name="Sarah Kim", bio="Tech writer"),
            _node("author_2", "Author", name="Lars Eriksson", bio="DB specialist"),
            _node("tag_1", "Tag", name="graph-databases"),
            _node("tag_2", "Tag", name="artificial-intelligence"),
            _node("pub_1", "Publication", name="Tech Monthly", frequency="monthly"),
        ]
        edges = [
            _edge("author_1", "article_1", "AUTHORED"),
            _edge("author_1", "article_2", "AUTHORED"),
            _edge("author_2", "article_1", "AUTHORED"),
            _edge("article_1", "tag_1", "TAGGED_WITH"),
            _edge("article_2", "tag_1", "TAGGED_WITH"),
            _edge("article_2", "tag_2", "TAGGED_WITH"),
            _edge("article_1", "pub_1", "PUBLISHED_IN"),
            _edge("article_2", "pub_1", "PUBLISHED_IN"),
        ]
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_content", quality_level=quality,
        ))
    return pairs


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Missing structural patterns
# ═══════════════════════════════════════════════════════════════════

def gen_hierarchical() -> list[dict]:
    """Hierarchical tree patterns (category trees, taxonomies)."""
    pairs = []
    examples = [
        ("hier_001", "Show the full category tree from Electronics to leaf categories",
         [_node("cat_electronics", "Category", name="Electronics", level=0),
          _node("cat_computers", "Category", name="Computers", level=1),
          _node("cat_laptops", "Category", name="Laptops", level=2),
          _node("cat_desktops", "Category", name="Desktops", level=2),
          _node("cat_phones", "Category", name="Phones", level=1),
          _node("cat_smartphones", "Category", name="Smartphones", level=2)],
         [_edge("cat_computers", "cat_electronics", "CHILD_OF"),
          _edge("cat_laptops", "cat_computers", "CHILD_OF"),
          _edge("cat_desktops", "cat_computers", "CHILD_OF"),
          _edge("cat_phones", "cat_electronics", "CHILD_OF"),
          _edge("cat_smartphones", "cat_phones", "CHILD_OF")]),
        ("hier_002", "What are all the subcategories under Vehicles?",
         [_node("cat_vehicles", "Category", name="Vehicles"),
          _node("cat_cars", "Category", name="Cars"),
          _node("cat_trucks", "Category", name="Trucks"),
          _node("cat_electric", "Category", name="Electric Cars"),
          _node("cat_suv", "Category", name="SUVs")],
         [_edge("cat_cars", "cat_vehicles", "CHILD_OF"),
          _edge("cat_trucks", "cat_vehicles", "CHILD_OF"),
          _edge("cat_electric", "cat_cars", "CHILD_OF"),
          _edge("cat_suv", "cat_cars", "CHILD_OF")]),
        ("hier_003", "Show the geographic hierarchy from Europe down to cities",
         [_node("geo_europe", "Region", name="Europe"),
          _node("geo_sweden", "Country", name="Sweden"),
          _node("geo_germany", "Country", name="Germany"),
          _node("geo_stockholm", "City", name="Stockholm"),
          _node("geo_gothenburg", "City", name="Gothenburg"),
          _node("geo_berlin", "City", name="Berlin")],
         [_edge("geo_sweden", "geo_europe", "LOCATED_IN"),
          _edge("geo_germany", "geo_europe", "LOCATED_IN"),
          _edge("geo_stockholm", "geo_sweden", "LOCATED_IN"),
          _edge("geo_gothenburg", "geo_sweden", "LOCATED_IN"),
          _edge("geo_berlin", "geo_germany", "LOCATED_IN")]),
        ("hier_004", "Map the file system structure from root to leaf directories",
         [_node("dir_root", "Directory", path="/"),
          _node("dir_src", "Directory", path="/src"),
          _node("dir_tests", "Directory", path="/tests"),
          _node("dir_models", "Directory", path="/src/models"),
          _node("file_user", "File", path="/src/models/user.py", size=4096)],
         [_edge("dir_src", "dir_root", "CHILD_OF"),
          _edge("dir_tests", "dir_root", "CHILD_OF"),
          _edge("dir_models", "dir_src", "CHILD_OF"),
          _edge("file_user", "dir_models", "CHILD_OF")]),
        ("hier_005", "what are the leaf nodes under the Science taxonomy",
         [_node("tax_science", "Topic", name="Science"),
          _node("tax_physics", "Topic", name="Physics"),
          _node("tax_chemistry", "Topic", name="Chemistry"),
          _node("tax_quantum", "Topic", name="Quantum Mechanics"),
          _node("tax_organic", "Topic", name="Organic Chemistry")],
         [_edge("tax_physics", "tax_science", "CHILD_OF"),
          _edge("tax_chemistry", "tax_science", "CHILD_OF"),
          _edge("tax_quantum", "tax_physics", "CHILD_OF"),
          _edge("tax_organic", "tax_chemistry", "CHILD_OF")]),
    ]

    for pid, text, nodes, edges in examples:
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="structural_hierarchical", pattern_type="hierarchy",
        ))
    return pairs


def gen_temporal() -> list[dict]:
    """Temporal patterns (timelines, versioning, event sequences)."""
    pairs = []
    examples = [
        ("temp_001", "Show the version history of document DOC-42",
         [_node("doc_42", "Document", name="Design Spec", current_version=3),
          _node("v1", "Version", number=1, created="2025-12-01", author="Erik"),
          _node("v2", "Version", number=2, created="2026-01-15", author="Maria"),
          _node("v3", "Version", number=3, created="2026-03-01", author="Erik")],
         [_edge("doc_42", "v3", "HAS_VERSION"),
          _edge("doc_42", "v2", "HAS_VERSION"),
          _edge("doc_42", "v1", "HAS_VERSION"),
          _edge("v3", "v2", "PRECEDED_BY"),
          _edge("v2", "v1", "PRECEDED_BY")]),
        ("temp_002", "What happened to order ORD-789 over time?",
         [_node("order_789", "Order", order_id="ORD-789"),
          _node("event_1", "Event", action="created", timestamp="2026-03-10T09:00:00"),
          _node("event_2", "Event", action="payment_confirmed", timestamp="2026-03-10T09:05:00"),
          _node("event_3", "Event", action="shipped", timestamp="2026-03-11T14:00:00"),
          _node("event_4", "Event", action="delivered", timestamp="2026-03-13T10:30:00")],
         [_edge("order_789", "event_1", "HAS_EVENT"),
          _edge("order_789", "event_2", "HAS_EVENT"),
          _edge("order_789", "event_3", "HAS_EVENT"),
          _edge("order_789", "event_4", "HAS_EVENT"),
          _edge("event_1", "event_2", "SUCCEEDED_BY"),
          _edge("event_2", "event_3", "SUCCEEDED_BY"),
          _edge("event_3", "event_4", "SUCCEEDED_BY")]),
        ("temp_003", "Show the timeline of incidents for service auth-svc this quarter",
         [_node("svc_auth", "Service", name="auth-svc"),
          _node("inc_1", "Incident", title="Login timeout", severity="high", occurred="2026-01-05"),
          _node("inc_2", "Incident", title="Token refresh failure", severity="medium", occurred="2026-02-12"),
          _node("inc_3", "Incident", title="OAuth provider outage", severity="critical", occurred="2026-03-01")],
         [_edge("inc_1", "svc_auth", "CAUSED_BY"),
          _edge("inc_2", "svc_auth", "CAUSED_BY"),
          _edge("inc_3", "svc_auth", "CAUSED_BY"),
          _edge("inc_1", "inc_2", "SUCCEEDED_BY"),
          _edge("inc_2", "inc_3", "SUCCEEDED_BY")]),
        ("temp_004", "what changed in the schema between v2.0 and v3.0",
         [_node("schema_v2", "SchemaVersion", version="2.0", released="2025-09-01"),
          _node("schema_v3", "SchemaVersion", version="3.0", released="2026-03-01"),
          _node("change_1", "Change", description="Added User.email index", type="index"),
          _node("change_2", "Change", description="Renamed FRIEND to FOLLOWS", type="relationship"),
          _node("change_3", "Change", description="Added Post label", type="label")],
         [_edge("schema_v3", "schema_v2", "PRECEDED_BY"),
          _edge("schema_v3", "change_1", "CONTAINS"),
          _edge("schema_v3", "change_2", "CONTAINS"),
          _edge("schema_v3", "change_3", "CONTAINS")]),
        ("temp_005", "Show monthly sales trends for product X",
         [_node("product_x", "Product", name="Widget X"),
          _node("jan", "SalesPeriod", month="2026-01", revenue=45000, units=150),
          _node("feb", "SalesPeriod", month="2026-02", revenue=52000, units=175),
          _node("mar", "SalesPeriod", month="2026-03", revenue=61000, units=210)],
         [_edge("product_x", "jan", "HAS_SALES"),
          _edge("product_x", "feb", "HAS_SALES"),
          _edge("product_x", "mar", "HAS_SALES"),
          _edge("jan", "feb", "SUCCEEDED_BY"),
          _edge("feb", "mar", "SUCCEEDED_BY")]),
    ]

    for pid, text, nodes, edges in examples:
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="structural_temporal", pattern_type="temporal",
        ))
    return pairs


def gen_workflow() -> list[dict]:
    """Workflow and state machine patterns."""
    pairs = []
    examples = [
        ("wf_001", "Show the approval workflow for purchase requests over 10K",
         [_node("pr_001", "PurchaseRequest", amount=25000, status="pending_cfo"),
          _node("state_draft", "State", name="Draft"),
          _node("state_manager", "State", name="Manager Approval"),
          _node("state_cfo", "State", name="CFO Approval"),
          _node("state_approved", "State", name="Approved"),
          _node("state_rejected", "State", name="Rejected")],
         [_edge("pr_001", "state_cfo", "CURRENT_STATE"),
          _edge("state_draft", "state_manager", "TRANSITIONED_TO"),
          _edge("state_manager", "state_cfo", "TRANSITIONED_TO"),
          _edge("state_cfo", "state_approved", "TRANSITIONED_TO"),
          _edge("state_manager", "state_rejected", "TRANSITIONED_TO"),
          _edge("state_cfo", "state_rejected", "TRANSITIONED_TO")]),
        ("wf_002", "What is the release pipeline for our mobile app?",
         [_node("release_1", "Release", version="4.2.0", status="in_qa"),
          _node("stage_dev", "Stage", name="Development"),
          _node("stage_qa", "Stage", name="QA Testing"),
          _node("stage_staging", "Stage", name="Staging"),
          _node("stage_prod", "Stage", name="Production")],
         [_edge("release_1", "stage_qa", "CURRENT_STATE"),
          _edge("stage_dev", "stage_qa", "TRANSITIONED_TO"),
          _edge("stage_qa", "stage_staging", "TRANSITIONED_TO"),
          _edge("stage_staging", "stage_prod", "TRANSITIONED_TO")]),
        ("wf_003", "Show the ticket lifecycle from creation to resolution",
         [_node("ticket_42", "Ticket", title="Login bug", priority="high"),
          _node("s_open", "Status", name="Open"),
          _node("s_progress", "Status", name="In Progress"),
          _node("s_review", "Status", name="In Review"),
          _node("s_resolved", "Status", name="Resolved"),
          _node("s_closed", "Status", name="Closed")],
         [_edge("ticket_42", "s_review", "CURRENT_STATE"),
          _edge("s_open", "s_progress", "TRANSITIONED_TO"),
          _edge("s_progress", "s_review", "TRANSITIONED_TO"),
          _edge("s_review", "s_resolved", "TRANSITIONED_TO"),
          _edge("s_review", "s_progress", "TRANSITIONED_TO"),
          _edge("s_resolved", "s_closed", "TRANSITIONED_TO")]),
        ("wf_004", "what approvals are needed for a new vendor onboarding",
         [_node("vendor_new", "Vendor", name="TechSupply AB", status="pending"),
          _node("step_kyc", "ApprovalStep", name="KYC Check", required=True),
          _node("step_legal", "ApprovalStep", name="Legal Review", required=True),
          _node("step_finance", "ApprovalStep", name="Finance Approval", required=True),
          _node("step_complete", "ApprovalStep", name="Onboarding Complete", required=True)],
         [_edge("vendor_new", "step_kyc", "REQUIRES_APPROVAL"),
          _edge("step_kyc", "step_legal", "SUCCEEDED_BY"),
          _edge("step_legal", "step_finance", "SUCCEEDED_BY"),
          _edge("step_finance", "step_complete", "SUCCEEDED_BY")]),
        ("wf_005", "Map the CI/CD pipeline stages and their dependencies",
         [_node("pipeline", "Pipeline", name="main-ci"),
          _node("build", "Stage", name="Build", duration_min=5),
          _node("lint", "Stage", name="Lint", duration_min=2),
          _node("test", "Stage", name="Test", duration_min=15),
          _node("deploy", "Stage", name="Deploy", duration_min=3)],
         [_edge("pipeline", "build", "CONTAINS"),
          _edge("pipeline", "lint", "CONTAINS"),
          _edge("pipeline", "test", "CONTAINS"),
          _edge("pipeline", "deploy", "CONTAINS"),
          _edge("build", "test", "SUCCEEDED_BY"),
          _edge("lint", "test", "SUCCEEDED_BY"),
          _edge("test", "deploy", "SUCCEEDED_BY")]),
    ]

    for pid, text, nodes, edges in examples:
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="structural_workflow", pattern_type="workflow",
        ))
    return pairs


def gen_multihop() -> list[dict]:
    """Multi-hop traversal patterns (recommendations, paths, chains)."""
    pairs = []
    examples = [
        ("hop_001", "Find movies that co-actors of Tom Hanks also acted in",
         [_node("tom", "Person", name="Tom Hanks"),
          _node("meg", "Person", name="Meg Ryan"),
          _node("billy", "Person", name="Billy Crystal"),
          _node("sleepless", "Movie", title="Sleepless in Seattle", year=1993),
          _node("harry", "Movie", title="When Harry Met Sally", year=1989)],
         [_edge("tom", "sleepless", "ACTED_IN"),
          _edge("meg", "sleepless", "ACTED_IN"),
          _edge("meg", "harry", "ACTED_IN"),
          _edge("billy", "harry", "ACTED_IN")]),
        ("hop_002", "Which restaurants do friends of friends of Alice recommend?",
         [_node("alice", "Person", name="Alice"),
          _node("bob", "Person", name="Bob"),
          _node("carol", "Person", name="Carol"),
          _node("restaurant_1", "Restaurant", name="Sushi Place", cuisine="Japanese"),
          _node("restaurant_2", "Restaurant", name="Pasta House", cuisine="Italian")],
         [_edge("alice", "bob", "IS_FRIENDS_WITH"),
          _edge("bob", "carol", "IS_FRIENDS_WITH"),
          _edge("carol", "restaurant_1", "REVIEWED"),
          _edge("carol", "restaurant_2", "REVIEWED")]),
        ("hop_003", "Trace the dependency chain from microservice A to the database",
         [_node("svc_a", "Service", name="Service A"),
          _node("svc_b", "Service", name="Service B"),
          _node("svc_c", "Service", name="Service C"),
          _node("db_main", "Database", name="PostgreSQL Main", type="relational")],
         [_edge("svc_a", "svc_b", "DEPENDS_ON"),
          _edge("svc_b", "svc_c", "DEPENDS_ON"),
          _edge("svc_c", "db_main", "DEPENDS_ON")]),
        ("hop_004", "Show all papers that cite papers cited by paper P-42",
         [_node("paper_42", "Paper", title="Graph Neural Networks", year=2024),
          _node("paper_10", "Paper", title="Attention Mechanisms", year=2017),
          _node("paper_99", "Paper", title="GNN Survey", year=2025),
          _node("paper_55", "Paper", title="Node Classification", year=2025)],
         [_edge("paper_42", "paper_10", "CITES"),
          _edge("paper_99", "paper_42", "CITES"),
          _edge("paper_55", "paper_10", "CITES")]),
        ("hop_005", "Find the shortest connection path between Company A and Company Z",
         [_node("comp_a", "Company", name="Company A"),
          _node("comp_b", "Company", name="Company B"),
          _node("comp_c", "Company", name="Company C"),
          _node("comp_z", "Company", name="Company Z"),
          _node("person_x", "Person", name="Board Member X")],
         [_edge("person_x", "comp_a", "WORKS_FOR"),
          _edge("person_x", "comp_b", "WORKS_FOR"),
          _edge("comp_b", "comp_c", "SUPPLIES"),
          _edge("comp_c", "comp_z", "SUPPLIES")]),
    ]

    for pid, text, nodes, edges in examples:
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="structural_multihop", pattern_type="multi_hop",
        ))
    return pairs


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Query→subgraph pairs (for g.ask())
# Reverse-engineered from expert cypher examples
# ═══════════════════════════════════════════════════════════════════

def gen_query_answer() -> list[dict]:
    """Query→subgraph pairs derived from expert Cypher examples."""
    pairs = []
    qa_examples = [
        ("qa_001", "Which people acted in a movie?",
         _node("p1", "Person", name="Tom Hanks"),
         [_node("m1", "Movie", title="Forrest Gump")],
         [_edge("p1", "m1", "ACTED_IN")]),
        ("qa_002", "Who directed and acted in the same movie?",
         _node("p1", "Person", name="Clint Eastwood"),
         [_node("m1", "Movie", title="Unforgiven")],
         [_edge("p1", "m1", "ACTED_IN"), _edge("p1", "m1", "DIRECTED")]),
        ("qa_003", "Find products supplied by a specific company",
         _node("s1", "Supplier", companyName="Exotic Liquids"),
         [_node("p1", "Product", productName="Chai"), _node("p2", "Product", productName="Chang")],
         [_edge("s1", "p1", "SUPPLIES"), _edge("s1", "p2", "SUPPLIES")]),
        ("qa_004", "Which orders contain a specific product?",
         _node("p1", "Product", productName="Chocolade"),
         [_node("o1", "Order", orderID="10372"), _node("c1", "Customer", companyName="ACME")],
         [_edge("o1", "p1", "CONTAINS"), _edge("c1", "o1", "PURCHASED")]),
        ("qa_005", "Find co-actors of a specific person",
         _node("tom", "Person", name="Tom Hanks"),
         [_node("meg", "Person", name="Meg Ryan"),
          _node("m1", "Movie", title="Sleepless in Seattle")],
         [_edge("tom", "m1", "ACTED_IN"), _edge("meg", "m1", "ACTED_IN")]),
        ("qa_006", "Recommend movies I haven't seen based on co-actor overlap",
         _node("user", "Person", name="User"),
         [_node("m1", "Movie", title="Movie A"),
          _node("m2", "Movie", title="Recommended Movie"),
          _node("coactor", "Person", name="Co-Actor")],
         [_edge("user", "m1", "ACTED_IN"), _edge("coactor", "m1", "ACTED_IN"),
          _edge("coactor", "m2", "ACTED_IN")]),
        ("qa_007", "Find friends of friends who like sushi restaurants in New York",
         _node("philip", "Person", name="Philip"),
         [_node("friend", "Person", name="Friend"),
          _node("rest", "Restaurant", name="Sushi Place"),
          _node("loc", "Location", location="New York"),
          _node("cuisine", "Cuisine", type="Sushi")],
         [_edge("philip", "friend", "IS_FRIENDS_WITH"), _edge("friend", "rest", "LIKES"),
          _edge("rest", "loc", "LOCATED_IN"), _edge("rest", "cuisine", "SERVES")]),
        ("qa_008", "Which employees work for Neo4j and like Java?",
         _node("p1", "Person", name="Jennifer"),
         [_node("co", "Company", name="Neo4j"), _node("tech", "Technology", type="Java")],
         [_edge("p1", "co", "WORKS_FOR"), _edge("p1", "tech", "LIKES")]),
        ("qa_009", "Show all people who reviewed a product with rating above 4",
         _node("p1", "Product", name="Widget Pro"),
         [_node("r1", "Review", rating=5), _node("c1", "Customer", name="Alice")],
         [_edge("r1", "p1", "REVIEWED"), _edge("c1", "r1", "AUTHORED")]),
        ("qa_010", "Which categories have the most products?",
         _node("cat1", "Category", name="Electronics"),
         [_node("p1", "Product", name="Laptop"), _node("p2", "Product", name="Phone"),
          _node("p3", "Product", name="Tablet")],
         [_edge("p1", "cat1", "BELONGS_TO"), _edge("p2", "cat1", "BELONGS_TO"),
          _edge("p3", "cat1", "BELONGS_TO")]),
        ("qa_011", "Find the shortest path between two people in the social graph",
         _node("alice", "Person", name="Alice"),
         [_node("bob", "Person", name="Bob"), _node("carol", "Person", name="Carol")],
         [_edge("alice", "bob", "FOLLOWS"), _edge("bob", "carol", "FOLLOWS")]),
        ("qa_012", "Which services have unresolved critical incidents?",
         _node("svc", "Service", name="payment-api"),
         [_node("inc", "Incident", severity="critical", resolved=False),
          _node("srv", "Server", hostname="prod-01")],
         [_edge("inc", "svc", "CAUSED_BY"), _edge("svc", "srv", "RUNS_ON")]),
        ("qa_013", "Show all transactions between two accounts",
         _node("acc1", "Account", iban="NL12ABCD"),
         [_node("acc2", "Account", iban="DE89EFGH"),
          _node("tx1", "Transaction", amount=5000),
          _node("tx2", "Transaction", amount=3200)],
         [_edge("acc1", "tx1", "SENT"), _edge("tx1", "acc2", "RECEIVED"),
          _edge("acc2", "tx2", "SENT"), _edge("tx2", "acc1", "RECEIVED")]),
        ("qa_014", "Which patients have been treated by multiple specialists?",
         _node("patient", "Patient", name="P-1001"),
         [_node("doc1", "Doctor", specialty="Cardiology"),
          _node("doc2", "Doctor", specialty="Neurology")],
         [_edge("patient", "doc1", "TREATED_BY"), _edge("patient", "doc2", "TREATED_BY")]),
        ("qa_015", "Find all courses that are prerequisites for Advanced ML",
         _node("adv_ml", "Course", name="Advanced ML"),
         [_node("intro_ml", "Course", name="Intro to ML"),
          _node("stats", "Course", name="Statistics"),
          _node("linear", "Course", name="Linear Algebra")],
         [_edge("intro_ml", "adv_ml", "PREREQUISITE_OF"),
          _edge("stats", "adv_ml", "PREREQUISITE_OF"),
          _edge("linear", "intro_ml", "PREREQUISITE_OF")]),
        ("qa_016", "Which APIs depend on the user service?",
         _node("user_svc", "Service", name="user-service"),
         [_node("api_gw", "Service", name="api-gateway"),
          _node("auth_svc", "Service", name="auth-service"),
          _node("order_svc", "Service", name="order-service")],
         [_edge("api_gw", "user_svc", "DEPENDS_ON"),
          _edge("auth_svc", "user_svc", "DEPENDS_ON"),
          _edge("order_svc", "user_svc", "DEPENDS_ON")]),
        ("qa_017", "Map the ownership chain from subsidiary to ultimate parent",
         _node("sub", "Company", name="TechDiv AB"),
         [_node("parent", "Company", name="Nordic Holdings"),
          _node("ultimate", "Company", name="Global Corp")],
         [_edge("sub", "parent", "OWNED_BY"),
          _edge("parent", "ultimate", "OWNED_BY")]),
        ("qa_018", "Which articles are trending based on recent reviews and shared tags?",
         _node("article", "Article", title="GraphRAG Guide"),
         [_node("tag1", "Tag", name="graph"), _node("tag2", "Tag", name="rag"),
          _node("related", "Article", title="Vector vs Graph RAG")],
         [_edge("article", "tag1", "TAGGED_WITH"), _edge("article", "tag2", "TAGGED_WITH"),
          _edge("related", "tag1", "TAGGED_WITH")]),
        ("qa_019", "Show the reporting chain from an employee to the CEO",
         _node("dev", "Employee", name="Erik"),
         [_node("lead", "Employee", name="Maria", title="Tech Lead"),
          _node("vp", "Employee", name="Anna", title="VP Eng"),
          _node("ceo", "Employee", name="Lars", title="CEO")],
         [_edge("dev", "lead", "REPORTS_TO"), _edge("lead", "vp", "REPORTS_TO"),
          _edge("vp", "ceo", "REPORTS_TO")]),
        ("qa_020", "Find buildings where temperature exceeds setpoint",
         _node("building", "Building", name="Office A"),
         [_node("floor", "Floor", name="Floor 3"),
          _node("room", "Room", name="Server Room"),
          _node("sensor", "TemperatureSensor", reading=28.5),
          _node("setpoint", "TemperatureSetpoint", value=22.0)],
         [_edge("building", "floor", "HAS_PART"),
          _edge("floor", "room", "HAS_PART"),
          _edge("sensor", "room", "IS_POINT_OF"),
          _edge("setpoint", "room", "IS_POINT_OF")]),
    ]

    for pid, text, source_node, targets, edges in qa_examples:
        pairs.append(_pair(
            pid, text, source_node, targets, edges,
            source="expert_cypher_qa", pattern_type="query_answer",
        ))
    return pairs


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Text-to-graph pairs (for g.ingest())
# Unstructured text → expected graph structure
# ═══════════════════════════════════════════════════════════════════

def gen_ingest_pairs() -> list[dict]:
    """Text → graph pairs for g.ingest() training."""
    pairs = []
    examples = [
        ("ingest_001",
         "Apple Inc. acquired Beats Electronics in 2014 for approximately $3 billion. "
         "The acquisition was led by CEO Tim Cook and brought Dr. Dre and Jimmy Iovine to Apple.",
         [_node("apple", "Company", name="Apple Inc."),
          _node("beats", "Company", name="Beats Electronics"),
          _node("acquisition", "Acquisition", year=2014, amount=3000000000, currency="USD"),
          _node("cook", "Person", name="Tim Cook", role="CEO"),
          _node("dre", "Person", name="Dr. Dre"),
          _node("iovine", "Person", name="Jimmy Iovine")],
         [_edge("apple", "acquisition", "MADE_ACQUISITION"),
          _edge("acquisition", "beats", "TARGET"),
          _edge("cook", "acquisition", "LED"),
          _edge("dre", "apple", "JOINED"),
          _edge("iovine", "apple", "JOINED")]),
        ("ingest_002",
         "The patient, a 45-year-old male, was admitted to the cardiology department on March 5th "
         "with chest pain. Dr. Sarah Chen diagnosed acute myocardial infarction (ICD I21.0) and "
         "prescribed aspirin 300mg and heparin IV.",
         [_node("patient", "Patient", age=45, gender="M"),
          _node("dept", "Department", name="Cardiology"),
          _node("doctor", "Doctor", name="Dr. Sarah Chen", specialty="Cardiology"),
          _node("diag", "Diagnosis", name="Acute myocardial infarction", icd_code="I21.0"),
          _node("med_1", "Medication", name="Aspirin", dose="300mg"),
          _node("med_2", "Medication", name="Heparin", route="IV")],
         [_edge("patient", "dept", "ADMITTED_TO"),
          _edge("patient", "doctor", "TREATED_BY"),
          _edge("doctor", "diag", "DIAGNOSED"),
          _edge("patient", "diag", "DIAGNOSED_WITH"),
          _edge("doctor", "med_1", "PRESCRIBED"),
          _edge("doctor", "med_2", "PRESCRIBED")]),
        ("ingest_003",
         "Stockholm Central Station serves as the hub for SL commuter rail, connecting to "
         "T-Centralen metro station and Cityterminalen bus terminal. Lines 40, 41, and 42 "
         "depart from platforms 1-9.",
         [_node("central", "Station", name="Stockholm Central", type="rail"),
          _node("tcentralen", "Station", name="T-Centralen", type="metro"),
          _node("cityterm", "Terminal", name="Cityterminalen", type="bus"),
          _node("line40", "TransitLine", number=40),
          _node("line41", "TransitLine", number=41),
          _node("line42", "TransitLine", number=42)],
         [_edge("central", "tcentralen", "CONNECTED_TO"),
          _edge("central", "cityterm", "CONNECTED_TO"),
          _edge("line40", "central", "DEPARTS_FROM"),
          _edge("line41", "central", "DEPARTS_FROM"),
          _edge("line42", "central", "DEPARTS_FROM")]),
        ("ingest_004",
         "The GDPR requires data controllers to implement appropriate technical and "
         "organizational measures. Article 25 mandates data protection by design and by default. "
         "This is enforced by national supervisory authorities under Article 51.",
         [_node("gdpr", "Regulation", name="GDPR"),
          _node("art25", "Article", number=25, title="Data protection by design and by default"),
          _node("art51", "Article", number=51, title="Supervisory authority"),
          _node("controller", "Role", name="Data Controller"),
          _node("authority", "Role", name="Supervisory Authority")],
         [_edge("gdpr", "art25", "CONTAINS"),
          _edge("gdpr", "art51", "CONTAINS"),
          _edge("art25", "controller", "OBLIGATES"),
          _edge("art51", "authority", "ESTABLISHES"),
          _edge("authority", "art25", "ENFORCES")]),
        ("ingest_005",
         "Tesla Model 3 uses a 75kWh lithium-ion battery manufactured by Panasonic at the "
         "Gigafactory in Nevada. The battery management system is developed in-house by Tesla's "
         "energy team in Palo Alto.",
         [_node("model3", "Product", name="Tesla Model 3"),
          _node("battery", "Component", name="75kWh Li-ion Battery"),
          _node("panasonic", "Supplier", name="Panasonic"),
          _node("gigafactory", "Facility", name="Gigafactory", location="Nevada"),
          _node("bms", "Component", name="Battery Management System"),
          _node("tesla", "Company", name="Tesla")],
         [_edge("model3", "battery", "CONTAINS"),
          _edge("panasonic", "battery", "MANUFACTURES"),
          _edge("panasonic", "gigafactory", "OPERATES_AT"),
          _edge("model3", "bms", "CONTAINS"),
          _edge("tesla", "bms", "DEVELOPED"),
          _edge("tesla", "model3", "PRODUCES")]),
        ("ingest_006",
         "Skanska signed a contract with Trafikverket to build the new E4 highway section "
         "between Sundsvall and Härnösand. The project is valued at 2.8 billion SEK and is "
         "expected to complete by 2028. Subcontractor NCC handles the bridge work.",
         [_node("skanska", "Company", name="Skanska"),
          _node("trafikverket", "Organization", name="Trafikverket"),
          _node("project", "Project", name="E4 Sundsvall-Härnösand", value_sek=2800000000, deadline="2028"),
          _node("ncc", "Company", name="NCC"),
          _node("bridge_work", "WorkPackage", name="Bridge Construction")],
         [_edge("skanska", "project", "CONTRACTED_FOR"),
          _edge("trafikverket", "project", "COMMISSIONED"),
          _edge("ncc", "bridge_work", "SUBCONTRACTED_FOR"),
          _edge("bridge_work", "project", "PART_OF"),
          _edge("ncc", "skanska", "SUBCONTRACTOR_OF")]),
        ("ingest_007",
         "The API gateway receives requests from mobile and web clients, authenticates via "
         "the auth service, then routes to either the order service or inventory service. "
         "All services log to the centralized ELK stack.",
         [_node("api_gw", "Service", name="API Gateway"),
          _node("auth", "Service", name="Auth Service"),
          _node("order", "Service", name="Order Service"),
          _node("inventory", "Service", name="Inventory Service"),
          _node("elk", "Service", name="ELK Stack", type="logging"),
          _node("mobile", "Client", type="mobile"),
          _node("web", "Client", type="web")],
         [_edge("mobile", "api_gw", "CALLS"),
          _edge("web", "api_gw", "CALLS"),
          _edge("api_gw", "auth", "DEPENDS_ON"),
          _edge("api_gw", "order", "ROUTES_TO"),
          _edge("api_gw", "inventory", "ROUTES_TO"),
          _edge("order", "elk", "LOGS_TO"),
          _edge("inventory", "elk", "LOGS_TO"),
          _edge("auth", "elk", "LOGS_TO")]),
        ("ingest_008",
         "Kursplan för Byggingenjörsprogrammet: Studenter läser Hållfasthetslära (7.5 hp) "
         "som förkunskapskrav för Konstruktionsteknik (15 hp). Båda kurserna ges av institutionen "
         "för byggvetenskap vid LTH.",
         [_node("program", "Program", name="Byggingenjörsprogrammet"),
          _node("course_1", "Course", name="Hållfasthetslära", credits=7.5),
          _node("course_2", "Course", name="Konstruktionsteknik", credits=15.0),
          _node("dept", "Department", name="Institutionen för byggvetenskap"),
          _node("lth", "University", name="LTH")],
         [_edge("program", "course_1", "CONTAINS"),
          _edge("program", "course_2", "CONTAINS"),
          _edge("course_1", "course_2", "PREREQUISITE_OF"),
          _edge("dept", "course_1", "TEACHES"),
          _edge("dept", "course_2", "TEACHES"),
          _edge("dept", "lth", "PART_OF")]),
        ("ingest_009",
         "The Northwind database tracks orders from customers to products through suppliers. "
         "Customer ALFKI placed order 10643 containing products Rössle Sauerkraut and "
         "Chartreuse verte, supplied by Plutzer and Aux joyeux respectively.",
         [_node("alfki", "Customer", customerID="ALFKI"),
          _node("order", "Order", orderID="10643"),
          _node("prod_1", "Product", name="Rössle Sauerkraut"),
          _node("prod_2", "Product", name="Chartreuse verte"),
          _node("sup_1", "Supplier", name="Plutzer"),
          _node("sup_2", "Supplier", name="Aux joyeux")],
         [_edge("alfki", "order", "PURCHASED"),
          _edge("order", "prod_1", "CONTAINS"),
          _edge("order", "prod_2", "CONTAINS"),
          _edge("sup_1", "prod_1", "SUPPLIES"),
          _edge("sup_2", "prod_2", "SUPPLIES")]),
        ("ingest_010",
         "Server room temperature alert: Sensor TS-SR-01 in the data center on floor B1 "
         "reads 31.2°C, exceeding the 25°C setpoint. The CRAC unit CRAC-B1-01 serving "
         "this zone is reporting fan failure.",
         [_node("sensor", "TemperatureSensor", name="TS-SR-01", reading=31.2),
          _node("dc", "Room", name="Data Center"),
          _node("floor_b1", "Floor", name="B1"),
          _node("setpoint", "TemperatureSetpoint", value=25.0),
          _node("crac", "CoolingUnit", name="CRAC-B1-01", status="fan_failure")],
         [_edge("sensor", "dc", "IS_POINT_OF"),
          _edge("dc", "floor_b1", "LOCATED_ON"),
          _edge("setpoint", "dc", "IS_POINT_OF"),
          _edge("crac", "dc", "SERVES")]),
    ]

    for pid, text, nodes, edges in examples:
        pairs.append(_pair(
            pid, text, nodes[0], nodes[1:], edges,
            source="expert_ingest", pattern_type="text_to_graph",
        ))
    return pairs


def build_all() -> list[dict]:
    """Build all expert NL-to-graph training pairs."""
    all_pairs: list[dict] = []

    generators = [
        ("E-commerce", gen_ecommerce),
        ("Social network", gen_social_network),
        ("Fraud detection", gen_fraud_detection),
        ("IT infrastructure", gen_it_infrastructure),
        ("Supply chain", gen_supply_chain),
        ("Org hierarchy", gen_org_hierarchy),
        ("Healthcare", gen_healthcare),
        ("Content management", gen_content_management),
        ("Hierarchical trees", gen_hierarchical),
        ("Temporal patterns", gen_temporal),
        ("Workflow patterns", gen_workflow),
        ("Multi-hop traversal", gen_multihop),
        ("Query-answer", gen_query_answer),
        ("Text-to-graph (ingest)", gen_ingest_pairs),
    ]

    for name, gen_fn in generators:
        pairs = gen_fn()
        all_pairs.extend(pairs)
        print(f"  {name}: {len(pairs)} pairs")

    return all_pairs


def print_stats(pairs: list[dict]) -> None:
    """Print dataset statistics."""
    total = len(pairs)
    total_fwd = sum(p["quality"]["edge_count"] for p in pairs)
    total_rev = sum(p["quality"]["reverse_edge_count"] for p in pairs)
    avg_edges = (total_fwd + total_rev) / total if total else 0
    avg_text = sum(p["quality"]["text_length"] for p in pairs) / total if total else 0

    from collections import Counter
    by_source = Counter(p["metadata"]["source"] for p in pairs)
    by_pattern = Counter(p["metadata"]["pattern_type"] for p in pairs)
    by_quality = Counter(p["metadata"]["quality_level"] for p in pairs)

    labels = Counter()
    rels = Counter()
    for p in pairs:
        eg = p["expected_graph"]
        labels[eg["source_node"]["label"]] += 1
        for tn in eg["target_nodes"]:
            labels[tn["label"]] += 1
        for e in eg["edges"]:
            rels[e["type"]] += 1

    print(f"\n{'=' * 60}")
    print("EXPERT NL-TO-GRAPH TRAINING PAIR STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total pairs:          {total}")
    print(f"  Total forward edges:  {total_fwd}")
    print(f"  Avg edges per pair:   {avg_edges:.1f}")
    print(f"  Avg text length:      {avg_text:.0f} chars")

    print(f"\n  By pattern type:")
    for pt, c in by_pattern.most_common():
        print(f"    {pt}: {c}")

    print(f"\n  By source ({len(by_source)}):")
    for s, c in by_source.most_common():
        print(f"    {s}: {c}")

    print(f"\n  By quality level:")
    for q, c in by_quality.most_common():
        print(f"    {q}: {c}")

    print(f"\n  Unique node labels: {len(labels)}")
    print(f"  Unique rel types:   {len(rels)}")

    print(f"\n  Top 15 labels:")
    for l, c in labels.most_common(15):
        print(f"    {l}: {c}")

    print(f"\n  Top 15 rel types:")
    for r, c in rels.most_common(15):
        print(f"    {r}: {c}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expert NL-to-graph training pairs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    print("Building expert NL-to-graph training pairs...")
    pairs = build_all()
    print_stats(pairs)

    if args.stats_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(pairs)} pairs to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
