"""Build fuzzy NL-to-graph training pairs from verified public schemas.

The hardest training tier: messy natural language input → production-ready
graph schema. Ground truth comes from verified Neo4j example datasets and
public graph benchmarks — NOT LLM-generated.

Each verified schema gets multiple NL prompts at varying quality levels:
- "clean": clear, well-structured request
- "casual": conversational, some ambiguity
- "messy": typos, fragments, stream-of-consciousness (real user input)

Training pair format (JSONL):
{
    "id": "movies_messy_1",
    "input_text": "running film studio, need to track who worked on what...",
    "metadata": {
        "domain": "nl_to_graph",
        "industry": "entertainment",
        "prompt_quality": "messy",
        "schema_source": "neo4j-movies",
        "source_url": "https://neo4j.com/docs/...",
        "verified": true
    },
    "expected_graph": {
        "source_node": {...},
        "target_nodes": [...],
        "edges": [...]
    },
    "quality": {
        "edge_count": 6,
        "node_count": 2,
        "text_length": 142,
        "verified_source": "neo4j_example_dataset"
    }
}

Usage:
    python data/scripts/build_nl_graph_pairs.py
    python data/scripts/build_nl_graph_pairs.py --stats-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = Path("data/training/nl_graph_pairs.jsonl")


# ---------------------------------------------------------------------------
# Verified schemas — every label, rel type, and property comes from official
# Neo4j example datasets or public graph benchmarks. No LLM generation.
# ---------------------------------------------------------------------------

SCHEMAS: list[dict] = [
    # ---- 1. Movies (Neo4j built-in) ----
    {
        "id": "movies",
        "industry": "entertainment",
        "source": "neo4j-movies",
        "source_url": "https://neo4j.com/docs/getting-started/appendix/tutorials/guide-cypher-basics/",
        "node_labels": {
            "Person": ["name", "born"],
            "Movie": ["title", "released", "tagline"],
        },
        "relationship_types": {
            "ACTED_IN": {"props": ["roles"], "from": "Person", "to": "Movie"},
            "DIRECTED": {"props": [], "from": "Person", "to": "Movie"},
            "PRODUCED": {"props": [], "from": "Person", "to": "Movie"},
            "WROTE": {"props": [], "from": "Person", "to": "Movie"},
            "REVIEWED": {"props": ["summary", "rating"], "from": "Person", "to": "Movie"},
            "FOLLOWS": {"props": [], "from": "Person", "to": "Person"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Build a movie database that tracks films, the people who acted in them, "
                    "directed them, produced them, and wrote them. Include review functionality "
                    "where people can rate and review movies. People should be able to follow "
                    "each other."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "I want to build something like IMDB. Movies, actors, directors, "
                    "producers, writers. People can review movies with ratings. "
                    "Also a social follow feature between users."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "running a film studio need to track who worked on what. actors directors "
                    "the whole thing. also want reviews like rotten tomatoes style with ratings "
                    "and ppl following each other"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "movie database, people act in direct produce write movies. "
                    "reviews with scores. social follows."
                ),
            },
        ],
    },
    # ---- 2. Northwind (Retail/Supply Chain) ----
    {
        "id": "northwind",
        "industry": "retail",
        "source": "neo4j-northwind",
        "source_url": (
            "https://neo4j.com/docs/getting-started/appendix/tutorials/"
            "guide-import-relational-and-etl/"
        ),
        "node_labels": {
            "Product": ["productID", "productName", "unitPrice"],
            "Order": ["orderID", "shipName"],
            "Category": ["categoryID", "categoryName", "description"],
            "Supplier": ["supplierID", "companyName"],
            "Employee": ["employeeID", "firstName", "lastName", "title"],
        },
        "relationship_types": {
            "CONTAINS": {
                "props": ["unitPrice", "quantity"],
                "from": "Order",
                "to": "Product",
            },
            "SOLD": {"props": [], "from": "Employee", "to": "Order"},
            "SUPPLIES": {"props": [], "from": "Supplier", "to": "Product"},
            "PART_OF": {"props": [], "from": "Product", "to": "Category"},
            "REPORTS_TO": {"props": [], "from": "Employee", "to": "Employee"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Design a retail order management system. Products belong to categories "
                    "and are supplied by suppliers. Employees process orders that contain "
                    "products with quantities and prices. Employees have a reporting hierarchy."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Need an e-commerce backend. Products, orders, categories, suppliers, "
                    "employees. Orders have line items with quantity and price. "
                    "Employees report to managers. Suppliers provide products."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "online store, products in categories, suppliers ship stuff. "
                    "employees sell orders with products and quantities. "
                    "manager hierarchy for the team"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "building supply chain tracking. who supplies what product, what category "
                    "its in, orders with line items, employee reporting structure"
                ),
            },
        ],
    },
    # ---- 3. POLE (Crime Investigation) ----
    {
        "id": "pole",
        "industry": "law_enforcement",
        "source": "neo4j-pole",
        "source_url": "https://github.com/neo4j-graph-examples/pole",
        "node_labels": {
            "Person": ["name", "surname", "nhs_no"],
            "Crime": ["type", "last_outcome"],
            "Location": ["address", "postcode", "longitude", "latitude"],
            "Officer": ["rank", "surname", "badge_no"],
        },
        "relationship_types": {
            "PARTY_TO": {"props": [], "from": "Person", "to": "Crime"},
            "INVESTIGATED_BY": {"props": [], "from": "Crime", "to": "Officer"},
            "OCCURRED_AT": {"props": [], "from": "Crime", "to": "Location"},
            "KNOWS": {"props": [], "from": "Person", "to": "Person"},
            "CURRENT_ADDRESS": {"props": [], "from": "Person", "to": "Location"},
            "FAMILY_REL": {"props": [], "from": "Person", "to": "Person"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Build a crime investigation graph. Track persons involved in crimes, "
                    "officers investigating cases, locations where crimes occurred. Map "
                    "relationships between persons including family and known associates. "
                    "Link persons to their addresses."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Police investigation tool. People connected to crimes, investigating "
                    "officers, crime locations. Need to see who knows who, family connections, "
                    "where people live."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "law enforcement graph, persons crimes officers locations. "
                    "who knows who, family ties, where crimes happened, "
                    "who investigated what, home addresses"
                ),
            },
        ],
    },
    # ---- 4. StackOverflow (Developer Community) ----
    {
        "id": "stackoverflow",
        "industry": "technology",
        "source": "neo4j-stackoverflow",
        "source_url": "https://github.com/neo4j-graph-examples/stackoverflow",
        "node_labels": {
            "User": ["display_name"],
            "Question": ["title", "body"],
            "Answer": ["is_accepted"],
            "Tag": ["name"],
            "Comment": ["text"],
        },
        "relationship_types": {
            "TAGGED": {"props": [], "from": "Question", "to": "Tag"},
            "ANSWERED": {"props": [], "from": "User", "to": "Question"},
            "PROVIDED": {"props": [], "from": "User", "to": "Answer"},
            "COMMENTED": {"props": [], "from": "User", "to": "Comment"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Design a Q&A platform like StackOverflow. Users ask questions and "
                    "provide answers. Questions are tagged with topics. Users can comment. "
                    "Answers can be marked as accepted."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Building a developer forum. Questions, answers, tags, comments. "
                    "Users post questions tagged by topic, other users answer them. "
                    "Best answer gets accepted."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "stackoverflow clone. users ask questions w tags, other users answer "
                    "and comment. accepted answers. need to track who posted what"
                ),
            },
        ],
    },
    # ---- 5. Healthcare Analytics (FDA Adverse Events) ----
    {
        "id": "healthcare",
        "industry": "healthcare",
        "source": "neo4j-healthcare-analytics",
        "source_url": "https://github.com/neo4j-graph-examples/healthcare-analytics",
        "node_labels": {
            "Manufacturer": ["manufacturerName"],
            "Case": ["gender", "age", "ageUnit"],
            "Drug": ["name"],
        },
        "relationship_types": {
            "REGISTERED": {"props": [], "from": "Manufacturer", "to": "Case"},
            "IS_PRIMARY_SUSPECT": {"props": [], "from": "Case", "to": "Drug"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Build a pharmacovigilance database tracking adverse drug events. "
                    "Pharmaceutical manufacturers register adverse event cases. Each case "
                    "involves a patient with demographics and identifies a primary suspect drug."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Drug safety monitoring system. Manufacturers report cases of bad "
                    "reactions. Each case has patient info (age, gender) and which drug "
                    "is suspected of causing it."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "pharma adverse events tracking. manufacturers, cases, drugs. "
                    "who made what drug, what went wrong, patient demographics. "
                    "FDA FAERS style"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "need to track drug side effects for compliance. manufacturer reports "
                    "a case, case links to suspected drug. patient age gender etc"
                ),
            },
        ],
    },
    # ---- 6. FinCEN (Anti-Money Laundering) ----
    {
        "id": "fincen",
        "industry": "finance",
        "source": "neo4j-fincen",
        "source_url": "https://github.com/neo4j-graph-examples/fincen",
        "node_labels": {
            "Filing": ["amount"],
            "FinancialEntity": ["name"],
            "Country": ["name"],
        },
        "relationship_types": {
            "ORIGINATOR": {"props": [], "from": "FinancialEntity", "to": "Filing"},
            "BENEFITS": {"props": [], "from": "FinancialEntity", "to": "Filing"},
            "BASED_IN": {"props": [], "from": "FinancialEntity", "to": "Country"},
            "TRANSFERRED": {"props": ["amount"], "from": "Filing", "to": "Filing"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Design an anti-money laundering investigation graph. Track suspicious "
                    "activity filings with monetary amounts. Entities can be originators or "
                    "beneficiaries of filings. Entities are associated with countries. "
                    "Filings can transfer to other filings."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "AML compliance tool. Track financial filings, who sends money, "
                    "who receives it, which countries are involved. Need to trace "
                    "money flows between filings."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "money laundering detection. entities file suspicious transactions, "
                    "originator and beneficiary, country links. trace transfers between filings"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "compliance team needs graph for tracking suspicious financial activity. "
                    "who sent what to whom, amounts, countries, chain of transfers"
                ),
            },
        ],
    },
    # ---- 7. Network Management (IT Infrastructure) ----
    {
        "id": "network",
        "industry": "it_infrastructure",
        "source": "neo4j-network-management",
        "source_url": "https://github.com/neo4j-graph-examples/network-management",
        "node_labels": {
            "DataCenter": ["name", "location"],
            "Rack": ["name"],
            "Router": ["name", "zone"],
            "Switch": ["ip", "rack"],
            "Machine": ["id", "name"],
            "Software": ["name"],
            "Process": ["name", "pid"],
        },
        "relationship_types": {
            "CONTAINS": {"props": [], "from": "DataCenter", "to": "Rack"},
            "HOLDS": {"props": [], "from": "Rack", "to": "Machine"},
            "ROUTES": {"props": [], "from": "Router", "to": "Router"},
            "PATCHED_TO": {"props": [], "from": "Switch", "to": "Machine"},
            "RUNS": {"props": [], "from": "Machine", "to": "Software"},
            "DEPENDS_ON": {"props": [], "from": "Software", "to": "Software"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Model an IT infrastructure graph. Data centers contain racks which "
                    "hold machines. Routers connect to each other. Switches connect to "
                    "machines. Machines run software that can depend on other software."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Network topology mapper. Data centers, racks, servers, routers, "
                    "switches. Need to track what runs where, software dependencies, "
                    "physical and network connectivity."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "IT infra graph. datacenters racks machines routers switches. "
                    "whats connected to what, what software runs on which machine, "
                    "dependency mapping"
                ),
            },
        ],
    },
    # ---- 8. Recommendations (Extended Movies) ----
    {
        "id": "recommendations",
        "industry": "entertainment",
        "source": "neo4j-recommendations",
        "source_url": "https://github.com/neo4j-graph-examples/recommendations",
        "node_labels": {
            "Movie": ["title", "year", "plot", "rating"],
            "User": ["name"],
            "Actor": ["name"],
            "Director": ["name"],
            "Genre": ["name"],
        },
        "relationship_types": {
            "ACTED_IN": {"props": [], "from": "Actor", "to": "Movie"},
            "DIRECTED": {"props": [], "from": "Director", "to": "Movie"},
            "IN_GENRE": {"props": [], "from": "Movie", "to": "Genre"},
            "RATED": {"props": ["rating"], "from": "User", "to": "Movie"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Build a movie recommendation engine. Movies have actors, directors, "
                    "and belong to genres. Users rate movies with numeric scores. "
                    "Use ratings and genre overlap to drive recommendations."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Netflix-style recommendation system. Movies, actors, directors, genres. "
                    "Users rate movies. Want to find similar movies by genre and cast overlap."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "recommendation engine for movies. actors directors genres ratings. "
                    "users rate films, suggest similar ones based on what they liked"
                ),
            },
        ],
    },
    # ---- 9. Academic (OGB MAG-inspired) ----
    {
        "id": "academic",
        "industry": "research",
        "source": "ogbn-mag",
        "source_url": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag",
        "node_labels": {
            "Paper": ["title", "year", "venue"],
            "Author": ["name"],
            "Institution": ["name"],
            "FieldOfStudy": ["name"],
        },
        "relationship_types": {
            "WRITES": {"props": [], "from": "Author", "to": "Paper"},
            "CITES": {"props": [], "from": "Paper", "to": "Paper"},
            "AFFILIATED_WITH": {"props": [], "from": "Author", "to": "Institution"},
            "HAS_TOPIC": {"props": [], "from": "Paper", "to": "FieldOfStudy"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Design an academic knowledge graph. Papers are written by authors who "
                    "are affiliated with institutions. Papers cite other papers and belong to "
                    "fields of study. Track publication venue and year."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Research paper database. Authors write papers, papers cite each other. "
                    "Authors work at universities. Papers tagged by research field. "
                    "Like Google Scholar but as a graph."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "academic citation graph. papers authors institutions fields. "
                    "who wrote what, who cites who, university affiliations, research topics"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "need to map research landscape. scientists at unis publishing papers "
                    "that reference each other across different fields"
                ),
            },
        ],
    },
    # ---- 10. Biomedical (OGB BioKG-inspired) ----
    {
        "id": "biomedical",
        "industry": "biomedical",
        "source": "ogbl-biokg",
        "source_url": "https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg",
        "node_labels": {
            "Disease": ["name"],
            "Protein": ["name"],
            "Drug": ["name"],
            "SideEffect": ["name"],
        },
        "relationship_types": {
            "TREATS": {"props": [], "from": "Drug", "to": "Disease"},
            "TARGETS": {"props": [], "from": "Drug", "to": "Protein"},
            "CAUSES": {"props": [], "from": "Drug", "to": "SideEffect"},
            "INTERACTS_WITH": {"props": [], "from": "Drug", "to": "Drug"},
            "IMPLICATED_IN": {"props": [], "from": "Protein", "to": "Disease"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Build a biomedical knowledge graph. Drugs treat diseases and target "
                    "specific proteins. Drugs can cause side effects and interact with other "
                    "drugs. Proteins are associated with diseases."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Drug discovery graph. Drugs, diseases, proteins, side effects. "
                    "Which drugs treat what, protein targets, drug interactions, "
                    "adverse effects tracking."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "pharma R&D knowledge graph. drugs target proteins cause side effects "
                    "treat diseases. drug drug interactions. protein disease associations"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "biotech startup needs to map drug mechanisms. what treats what, "
                    "side effects, protein targets, how drugs interact with each other"
                ),
            },
        ],
    },
    # ---- 11. Offshore Leaks (Investigative Journalism) ----
    {
        "id": "offshore",
        "industry": "finance",
        "source": "icij-offshoreleaks",
        "source_url": "https://github.com/neo4j-graph-examples/icij-offshoreleaks",
        "node_labels": {
            "Officer": ["name"],
            "OffshoreEntity": ["name", "jurisdiction"],
            "Intermediary": ["name"],
            "Address": ["address", "country"],
        },
        "relationship_types": {
            "OFFICER_OF": {"props": [], "from": "Officer", "to": "OffshoreEntity"},
            "INTERMEDIARY_OF": {"props": [], "from": "Intermediary", "to": "OffshoreEntity"},
            "REGISTERED_ADDRESS": {"props": [], "from": "OffshoreEntity", "to": "Address"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Design a corporate ownership investigation graph. Officers control "
                    "offshore entities. Intermediaries facilitate entity creation. "
                    "Entities have registered addresses in various jurisdictions."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Panama Papers style investigation. Shell companies, their officers, "
                    "intermediaries who set them up, registered addresses. "
                    "Need to trace ownership across jurisdictions."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "offshore company investigation. who owns what shell company, "
                    "intermediaries, registered addresses, jurisdictions. trace the money"
                ),
            },
        ],
    },
    # ---- 12. Fraud Detection (Banking) ----
    {
        "id": "fraud",
        "industry": "finance",
        "source": "neo4j-fraud-detection-pattern",
        "source_url": "https://neo4j.com/developer/graph-data-science/fraud-detection/",
        "node_labels": {
            "AccountHolder": ["name", "dateOfBirth"],
            "Account": ["accountNumber", "balance"],
            "Transaction": ["amount", "date", "type"],
            "PhoneNumber": ["number"],
            "SocialSecurityNumber": ["number"],
            "Address": ["street", "city", "state"],
        },
        "relationship_types": {
            "HAS_ACCOUNT": {"props": [], "from": "AccountHolder", "to": "Account"},
            "PERFORMED": {"props": [], "from": "Account", "to": "Transaction"},
            "TRANSFERRED_TO": {"props": [], "from": "Transaction", "to": "Account"},
            "HAS_PHONE": {"props": [], "from": "AccountHolder", "to": "PhoneNumber"},
            "HAS_SSN": {"props": [], "from": "AccountHolder", "to": "SocialSecurityNumber"},
            "HAS_ADDRESS": {"props": [], "from": "AccountHolder", "to": "Address"},
        },
        "prompts": [
            {
                "quality": "clean",
                "text": (
                    "Build a fraud detection graph for a bank. Account holders have accounts, "
                    "phone numbers, SSNs, and addresses. Accounts perform transactions that "
                    "transfer money to other accounts. Detect fraud rings by finding shared "
                    "identity attributes across account holders."
                ),
            },
            {
                "quality": "casual",
                "text": (
                    "Bank fraud detection. Customers, accounts, transactions. Track phone "
                    "numbers, SSNs, addresses per customer. Find fraud when multiple customers "
                    "share the same phone or SSN."
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "fraud ring detection for banking. accounts transactions transfers. "
                    "customers share phone numbers SSNs addresses — thats how you find "
                    "the rings. need to map it all"
                ),
            },
            {
                "quality": "messy",
                "text": (
                    "bank compliance, need to detect fraud. who has which accounts, "
                    "money transfers between accounts, shared identity signals like "
                    "same address same phone same SSN"
                ),
            },
        ],
    },
]


# Edge weight scheme for NL→graph pairs
EDGE_WEIGHTS: dict[str, float] = {
    "structural": 0.9,  # core domain relationships
    "identity": 0.8,  # shared-identity links (fraud)
    "hierarchical": 0.7,  # containment / reporting
    "social": 0.6,  # follows, knows
}


def schema_to_graph(schema: dict) -> dict:
    """Convert a verified schema into expected_graph format."""
    labels = schema["node_labels"]
    rels = schema["relationship_types"]

    # Build all nodes
    all_nodes: dict[str, dict] = {}
    for label, props in labels.items():
        node_id = f"{schema['id']}_{label.lower()}"
        all_nodes[label] = {
            "id": node_id,
            "label": label,
            "properties": {
                "label": label,
                "properties": props,
            },
        }

    # Source node is the first label (primary entity)
    primary_label = next(iter(labels.keys()))
    source_node = all_nodes[primary_label]

    # Target nodes = all labels except the primary (source_node)
    target_nodes = [all_nodes[label] for label in labels if label != primary_label]

    # Edges from relationship types
    edges = []
    for rel_type, rel_info in rels.items():
        from_label = rel_info["from"]
        to_label = rel_info["to"]
        edge = {
            "source": f"{schema['id']}_{from_label.lower()}",
            "target": f"{schema['id']}_{to_label.lower()}",
            "type": rel_type,
            "weight": 0.9,
        }
        if rel_info.get("props"):
            edge["relationship_properties"] = rel_info["props"]
        edges.append(edge)

    return {
        "source_node": source_node,
        "target_nodes": target_nodes,
        "edges": edges,
    }


def build_pairs() -> list[dict]:
    """Build training pairs from all schemas and their prompts."""
    pairs = []

    for schema in SCHEMAS:
        expected_graph = schema_to_graph(schema)
        num_labels = len(schema["node_labels"])
        num_rels = len(schema["relationship_types"])

        for i, prompt in enumerate(schema["prompts"]):
            pair_id = f"{schema['id']}_{prompt['quality']}_{i}"

            pair = {
                "id": pair_id,
                "input_text": prompt["text"],
                "metadata": {
                    "domain": "nl_to_graph",
                    "industry": schema["industry"],
                    "prompt_quality": prompt["quality"],
                    "schema_source": schema["source"],
                    "source_url": schema["source_url"],
                    "verified": True,
                },
                "expected_graph": expected_graph,
                "quality": {
                    "node_count": num_labels,
                    "edge_count": num_rels,
                    "text_length": len(prompt["text"]),
                    "prompt_quality": prompt["quality"],
                    "verified_source": "public_graph_schema",
                },
            }
            pairs.append(pair)

    return pairs


def print_stats(pairs: list[dict]) -> None:
    """Print dataset statistics."""
    total = len(pairs)
    if total == 0:
        print("No pairs generated.")
        return

    by_industry: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    by_schema: dict[str, int] = {}
    all_labels: set[str] = set()
    all_rels: set[str] = set()

    for p in pairs:
        ind = p["metadata"]["industry"]
        by_industry[ind] = by_industry.get(ind, 0) + 1
        q = p["metadata"]["prompt_quality"]
        by_quality[q] = by_quality.get(q, 0) + 1
        src = p["metadata"]["schema_source"]
        by_schema[src] = by_schema.get(src, 0) + 1

        eg = p["expected_graph"]
        all_labels.add(eg["source_node"]["label"])
        for tn in eg["target_nodes"]:
            all_labels.add(tn["label"])
        for e in eg["edges"]:
            all_rels.add(e["type"])

    text_lengths = [p["quality"]["text_length"] for p in pairs]
    avg_text = sum(text_lengths) / len(text_lengths)
    node_counts = [p["quality"]["node_count"] for p in pairs]
    avg_nodes = sum(node_counts) / len(node_counts)

    print(f"\n{'=' * 60}")
    print("NL-TO-GRAPH TRAINING PAIR STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total pairs:              {total}")
    print(f"  Unique schemas:           {len(by_schema)}")
    print(f"  Unique node labels:       {len(all_labels)}")
    print(f"  Unique rel types:         {len(all_rels)}")
    print(f"  Avg text length:          {avg_text:.0f} chars")
    print(f"  Avg nodes per schema:     {avg_nodes:.1f}")

    print("\n  By prompt quality:")
    for q, count in sorted(by_quality.items()):
        print(f"    {q}: {count}")

    print("\n  By industry:")
    for ind, count in sorted(by_industry.items(), key=lambda x: -x[1]):
        print(f"    {ind}: {count}")

    print("\n  By schema:")
    for src, count in sorted(by_schema.items()):
        print(f"    {src}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NL-to-graph training pairs from verified schemas"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics without writing output",
    )
    args = parser.parse_args()

    pairs = build_pairs()
    print_stats(pairs)

    if args.stats_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(pairs)} training pairs to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
