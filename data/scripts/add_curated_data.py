"""One-time script to add curated industry patterns and best practices."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1].parent / "src" / "gibsgraph" / "data"


def add_industry_patterns() -> int:
    """Append 8 industry modeling patterns."""
    patterns = [
        {
            "name": "Supply chain tracking",
            "description": "Model supply chain visibility with Supplier, Product, Warehouse, and Shipment nodes. Track procurement, inventory, and logistics as a connected graph for multi-supplier risk analysis and delivery tracking.",
            "when_to_use": "When you need end-to-end supply chain visibility: supplier sourcing, inventory management across warehouses, shipment tracking, and lead time analysis. Graph queries excel at multi-hop supplier dependency analysis.",
            "anti_pattern": "Do not model shipment details as properties on Product nodes. Shipments are events with their own lifecycle (status, carrier, dates) and should be separate nodes connected via SHIPPED_VIA relationships.",
            "cypher_examples": [
                "MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)-[:STORED_IN]->(w:Warehouse)\nWHERE w.location = $location\nRETURN s.name, p.name, w.location",
                "MATCH (p:Product)-[:SHIPPED_VIA]->(sh:Shipment)\nWHERE sh.status = 'delayed'\nRETURN p.sku, sh.carrier, sh.shipped_at",
            ],
            "node_labels": ["Supplier", "Product", "Warehouse", "Shipment"],
            "relationship_types": ["SUPPLIES", "STORED_IN", "SHIPPED_VIA"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "Social network graph",
            "description": "Model social interactions with Person, Post, and Community nodes. Directional FOLLOWS relationships enable feed algorithms, AUTHORED links content to creators, and MEMBER_OF supports community features.",
            "when_to_use": "When building social features: follower feeds, content recommendations, community detection, influence scoring. Graph traversal naturally handles friend-of-friend queries and mutual connection analysis.",
            "anti_pattern": "Do not store follower lists as arrays on Person nodes. FOLLOWS should be a relationship so you can traverse the social graph efficiently. Avoid bidirectional FRIENDS_WITH when the relationship is inherently directional.",
            "cypher_examples": [
                "MATCH (me:Person {id: $userId})-[:FOLLOWS]->(friend)-[:AUTHORED]->(post:Post)\nWHERE post.created_at > datetime() - duration('P7D')\nRETURN post ORDER BY post.created_at DESC LIMIT 20",
                "MATCH (a:Person)-[:FOLLOWS]->(mutual)<-[:FOLLOWS]-(b:Person)\nWHERE a.id = $userId AND NOT (a)-[:FOLLOWS]->(b)\nRETURN b.name, count(mutual) AS mutual_friends ORDER BY mutual_friends DESC",
            ],
            "node_labels": ["Person", "Post", "Community"],
            "relationship_types": ["FOLLOWS", "AUTHORED", "MEMBER_OF"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "Fraud detection graph",
            "description": "Model financial fraud detection with Customer, Account, Transaction, and Device nodes. Graph structure reveals fraud rings through shared devices, circular transaction chains, and unusual account relationships.",
            "when_to_use": "When detecting financial fraud: transaction monitoring, fraud ring identification, device fingerprint analysis, KYC/AML compliance. Graph queries find patterns that relational databases miss.",
            "anti_pattern": "Do not model transactions as properties on Account nodes. Each transaction is an event connecting sender and receiver accounts. Device should be a separate node, not a property, to detect device sharing across unrelated accounts.",
            "cypher_examples": [
                "MATCH (a1:Account)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(a2:Account)\nWHERE a1 <> a2 AND NOT (a1)<-[:OWNS]-(:Customer)-[:OWNS]->(a2)\nRETURN d.fingerprint, collect(DISTINCT a1.iban) + collect(DISTINCT a2.iban) AS suspicious_accounts",
                "MATCH path = (a:Account)-[:SENT]->(:Transaction)-[:RECEIVED]->(b:Account)-[:SENT]->(:Transaction)-[:RECEIVED]->(a)\nWHERE a.id = $accountId\nRETURN path",
            ],
            "node_labels": ["Customer", "Account", "Transaction", "Device"],
            "relationship_types": ["OWNS", "SENT", "RECEIVED", "USED_DEVICE"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "E-commerce product catalog",
            "description": "Model e-commerce with Customer, Product, Category, and Review nodes. PURCHASED captures buying behavior, BELONGS_TO organizes taxonomy, and REVIEWED enables recommendation engines based on graph similarity.",
            "when_to_use": "When building product recommendations, catalog browsing, or customer behavior analysis. Graph-based collaborative filtering outperforms table joins at scale.",
            "anti_pattern": "Do not flatten category hierarchy into a single level. Use CHILD_OF relationships between Category nodes for multi-level taxonomy. Do not store reviews as properties on Product.",
            "cypher_examples": [
                "MATCH (c:Customer)-[:PURCHASED]->(p:Product)<-[:PURCHASED]-(other:Customer)-[:PURCHASED]->(rec:Product)\nWHERE c.id = $customerId AND NOT (c)-[:PURCHASED]->(rec)\nRETURN rec.name, count(other) AS score ORDER BY score DESC LIMIT 10",
                "MATCH (p:Product)-[:BELONGS_TO]->(cat:Category)\nOPTIONAL MATCH (p)<-[r:REVIEWED]-()\nRETURN cat.name, count(DISTINCT p) AS products, avg(r.rating) AS avg_rating\nORDER BY products DESC",
            ],
            "node_labels": ["Customer", "Product", "Category", "Review"],
            "relationship_types": ["PURCHASED", "BELONGS_TO", "REVIEWED", "CHILD_OF"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "IT infrastructure dependency graph",
            "description": "Model IT infrastructure with Server, Service, Network, and Incident nodes. RUNS_ON and DEPENDS_ON relationships enable impact analysis when a server goes down.",
            "when_to_use": "When building CMDB, incident management, or change impact analysis. Graph traversal answers what breaks if this server goes down via dependency chain analysis.",
            "anti_pattern": "Do not model dependencies as a flat list on Service nodes. Dependencies are relationships that form chains. Only graph traversal can follow these chains efficiently.",
            "cypher_examples": [
                "MATCH (s:Server {id: $serverId})<-[:RUNS_ON]-(svc:Service)<-[:DEPENDS_ON*1..3]-(affected:Service)\nRETURN affected.name, length(shortestPath((s)<-[:RUNS_ON|DEPENDS_ON*]-(affected))) AS distance\nORDER BY distance",
                "MATCH (i:Incident)-[:CAUSED_BY]->(s:Service)-[:RUNS_ON]->(srv:Server)\nWHERE i.severity = 'critical' AND i.resolved_at IS NULL\nRETURN srv.name, collect(s.name) AS affected_services, count(i) AS open_incidents",
            ],
            "node_labels": ["Server", "Service", "Network", "Incident"],
            "relationship_types": ["RUNS_ON", "DEPENDS_ON", "CONNECTED_TO", "CAUSED_BY"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "Healthcare patient journey",
            "description": "Model patient care with Patient, Doctor, Diagnosis, and Treatment nodes. Graph queries identify treatment patterns, drug interactions via shared prescriptions, and care pathway compliance.",
            "when_to_use": "When building clinical decision support, patient outcome analysis, or treatment pathway optimization.",
            "anti_pattern": "Do not model diagnoses as string properties on Patient. Each diagnosis is an event with its own ICD code, date, and treating physician.",
            "cypher_examples": [
                "MATCH (p:Patient)-[:DIAGNOSED_WITH]->(d:Diagnosis)<-[:DIAGNOSED_WITH]-(other:Patient)-[:PRESCRIBED]->(t:Treatment)\nWHERE p.id = $patientId AND NOT (p)-[:PRESCRIBED]->(t)\nRETURN t.name, count(other) AS similar_patients ORDER BY similar_patients DESC",
                "MATCH (p:Patient)-[:TREATED_BY]->(doc:Doctor)\nMATCH (p)-[:DIAGNOSED_WITH]->(d:Diagnosis)\nWHERE d.icd_code STARTS WITH $icdPrefix\nRETURN doc.name, count(DISTINCT p) AS patients, collect(DISTINCT d.icd_code) AS diagnoses",
            ],
            "node_labels": ["Patient", "Doctor", "Diagnosis", "Treatment"],
            "relationship_types": ["TREATED_BY", "DIAGNOSED_WITH", "PRESCRIBED"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "Content management graph",
            "description": "Model content publishing with Article, Author, Tag, and Publication nodes. Graph queries find related articles via shared tags and co-authorship, and enable content gap analysis.",
            "when_to_use": "When building CMS, content recommendation, or editorial workflow.",
            "anti_pattern": "Do not store tags as a comma-separated string property. Each tag should be a node connected via TAGGED_WITH for tag co-occurrence queries.",
            "cypher_examples": [
                "MATCH (a:Article)-[:TAGGED_WITH]->(t:Tag)<-[:TAGGED_WITH]-(related:Article)\nWHERE a.id = $articleId AND related <> a\nRETURN related.title, count(t) AS shared_tags ORDER BY shared_tags DESC LIMIT 5",
                "MATCH (author:Author)-[:AUTHORED]->(a:Article)-[:TAGGED_WITH]->(t:Tag)\nRETURN author.name, collect(DISTINCT t.name) AS topics, count(a) AS articles\nORDER BY articles DESC",
            ],
            "node_labels": ["Article", "Author", "Tag", "Publication"],
            "relationship_types": ["AUTHORED", "TAGGED_WITH", "PUBLISHED_IN"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
        {
            "name": "Organizational hierarchy",
            "description": "Model corporate structure with Employee, Department, Role, and Project nodes. REPORTS_TO creates management chains, WORKS_IN links to departments, and ASSIGNED_TO connects people to projects.",
            "when_to_use": "When building org charts, resource planning, or project staffing tools. Graph traversal naturally handles who reports to whom chains.",
            "anti_pattern": "Do not model the reporting chain as a manager_id property. REPORTS_TO relationships allow arbitrary depth traversal. Do not flatten departments into a single level if sub-departments exist.",
            "cypher_examples": [
                "MATCH path = (e:Employee)-[:REPORTS_TO*]->(ceo:Employee)\nWHERE e.id = $employeeId AND NOT (ceo)-[:REPORTS_TO]->()\nRETURN [n IN nodes(path) | n.name] AS chain, length(path) AS levels",
                "MATCH (e:Employee)-[:WORKS_IN]->(d:Department)\nMATCH (e)-[:ASSIGNED_TO]->(p:Project)\nRETURN d.name, count(DISTINCT e) AS headcount, collect(DISTINCT p.name) AS projects",
            ],
            "node_labels": ["Employee", "Department", "Role", "Project"],
            "relationship_types": ["REPORTS_TO", "WORKS_IN", "ASSIGNED_TO", "HAS_ROLE"],
            "source_file": "curated/industry_patterns",
            "authority_level": 2,
        },
    ]

    path = DATA_DIR / "modeling_patterns.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for p in patterns:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(patterns)


def add_best_practices() -> int:
    """Append 10 best practices: 5 error recovery + 5 index guidance."""
    practices = [
        # Error recovery
        {
            "title": "Avoiding unintended cartesian products",
            "description": "Multiple disconnected MATCH clauses create a cartesian product of all matched rows. Always connect MATCH patterns or use WITH to scope variables. Instead of separate MATCH clauses for unrelated patterns, connect them through shared variables or use WITH between them.",
            "category": "cypher",
            "cypher_examples": [
                "MATCH (a:Person)-[:ACTED_IN]->(m:Movie)\nWITH a, collect(m) AS movies\nMATCH (d:Director)-[:DIRECTED]->(m2:Movie)\nWHERE m2 IN movies\nRETURN a.name, d.name"
            ],
            "source_file": "curated/error_recovery",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Fixing type mismatch in WHERE clauses",
            "description": "Cypher is type-strict in comparisons. Comparing a string property to an integer literal returns no results silently. Use toInteger(), toFloat(), or toString() to coerce types. Check property types when debugging empty results from WHERE filters.",
            "category": "cypher",
            "cypher_examples": [
                "MATCH (p:Product)\nWHERE toFloat(p.price) > $minPrice\nRETURN p.name, p.price"
            ],
            "source_file": "curated/error_recovery",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Eager aggregation invalidates bindings",
            "description": "Aggregation functions (count, collect, sum) in RETURN or WITH consume all previous bindings except the grouping keys. If you need to reference non-grouped variables after aggregation, include them in WITH before aggregating or re-MATCH after the WITH clause.",
            "category": "cypher",
            "cypher_examples": [
                "MATCH (p:Person)-[:ACTED_IN]->(m:Movie)\nWITH p, count(m) AS movie_count\nMATCH (p)-[:DIRECTED]->(d:Movie)\nRETURN p.name, movie_count, collect(d.title) AS directed"
            ],
            "source_file": "curated/error_recovery",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Using UNWIND for list parameters",
            "description": "When passing a list parameter to match multiple values, use UNWIND $list AS item before MATCH. Writing WHERE n.id IN $ids works for simple lookups but UNWIND is required when each list item needs its own processing pipeline or when combining with other MATCH patterns.",
            "category": "cypher",
            "cypher_examples": [
                "UNWIND $names AS name\nMATCH (p:Person {name: name})-[:ACTED_IN]->(m:Movie)\nRETURN name, collect(m.title) AS movies"
            ],
            "source_file": "curated/error_recovery",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Relationship direction matters for query results",
            "description": "Neo4j relationships are always directional. A query with (a)-[:KNOWS]->(b) will not match (a)<-[:KNOWS]-(b). If direction does not matter, use undirected pattern (a)-[:KNOWS]-(b). When a query returns 0 rows unexpectedly, check if the relationship direction in the pattern matches the data.",
            "category": "cypher",
            "cypher_examples": [
                "MATCH (a:Person)-[:KNOWS]-(b:Person)\nWHERE a.name = $name\nRETURN b.name"
            ],
            "source_file": "curated/error_recovery",
            "authority_level": 2,
            "quality_tier": "high",
        },
        # Index guidance
        {
            "title": "Use RANGE indexes for equality and range queries",
            "description": "RANGE indexes (the default in Neo4j 5.x) support equality, range comparisons, IN, STARTS WITH, and existence checks. Create RANGE indexes on properties used in WHERE clauses. For composite lookups, create a composite index on multiple properties.",
            "category": "performance",
            "cypher_examples": [
                "CREATE INDEX product_category_price IF NOT EXISTS FOR (p:Product) ON (p.category, p.price)"
            ],
            "source_file": "curated/index_guidance",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Use TEXT indexes for CONTAINS and regex queries",
            "description": "RANGE indexes do not support CONTAINS or regular expression queries. Create a TEXT index when your queries use CONTAINS, ENDS WITH, or regex matching with =~. TEXT indexes are backed by Lucene and handle full-text-like operations efficiently.",
            "category": "performance",
            "cypher_examples": [
                "CREATE TEXT INDEX article_content IF NOT EXISTS FOR (a:Article) ON (a.content)"
            ],
            "source_file": "curated/index_guidance",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Use VECTOR indexes for embedding similarity search",
            "description": "Neo4j 5.11+ supports native vector indexes for cosine, euclidean, or dot-product similarity. Store embeddings as float arrays on nodes and create a vector index with the correct dimensions. Use db.index.vector.queryNodes() for approximate nearest neighbor search.",
            "category": "performance",
            "cypher_examples": [
                "CREATE VECTOR INDEX article_embedding IF NOT EXISTS FOR (a:Article) ON (a.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
            ],
            "source_file": "curated/index_guidance",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "Composite indexes for multi-property lookups",
            "description": "When queries frequently filter on two or more properties of the same label, create a composite index. The order of properties matters: put the most selective property first. Composite indexes support prefix matching: an index on (status, created_at) also speeds up queries filtering only on status.",
            "category": "performance",
            "cypher_examples": [
                "CREATE INDEX order_status_date IF NOT EXISTS FOR (o:Order) ON (o.status, o.created_at)"
            ],
            "source_file": "curated/index_guidance",
            "authority_level": 2,
            "quality_tier": "high",
        },
        {
            "title": "When NOT to create an index",
            "description": "Do not index low-cardinality properties like booleans: the index scan returns too many rows to be useful. Do not index properties only used in RETURN but never in WHERE. Each index adds write overhead and memory usage. Use PROFILE to verify the planner actually uses your index.",
            "category": "performance",
            "cypher_examples": [
                "PROFILE MATCH (p:Product) WHERE p.category = $cat RETURN p"
            ],
            "source_file": "curated/index_guidance",
            "authority_level": 2,
            "quality_tier": "high",
        },
    ]

    path = DATA_DIR / "best_practices.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for p in practices:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(practices)


if __name__ == "__main__":
    n_patterns = add_industry_patterns()
    n_practices = add_best_practices()
    print(f"Added {n_patterns} industry patterns and {n_practices} best practices")
