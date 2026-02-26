"""Independent golden test fixtures for validation.

These schemas were NOT written with knowledge of the validator's internal
checks. They represent real-world Neo4j use cases sourced from public
Neo4j documentation and community examples.

Three categories:
  - KNOWN_GOOD: Real domain schemas that should score well
  - KNOWN_BAD: Plausible but structurally broken schemas
  - ADVERSARIAL: Designed to game the validator (should NOT score high)

RULE: This file must NEVER import from gibsgraph.training.validator or
gibsgraph.training.prompts. It only imports the data models.
"""

from gibsgraph.training.models import GraphSchema, NodeSchema, RelationshipSchema

# ---------------------------------------------------------------------------
# KNOWN GOOD — real Neo4j use cases from public examples
# ---------------------------------------------------------------------------


def supply_chain_schema() -> GraphSchema:
    """Supply chain tracking — from Neo4j's supply chain management examples.

    Source: Neo4j GraphAcademy supply chain course + Neo4j blog posts.
    """
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Supplier",
                properties=["id", "name", "country", "rating", "certified_since"],
                required_properties=["id", "name", "country"],
                description="Raw material or component supplier",
                justified_by="Supply chain visibility requires tracking every supplier node",
            ),
            NodeSchema(
                label="Product",
                properties=["id", "sku", "name", "category", "weight_kg"],
                required_properties=["id", "sku", "name"],
                description="Manufactured product or component",
                justified_by="Product traceability from raw material to finished good",
            ),
            NodeSchema(
                label="Warehouse",
                properties=["id", "location", "capacity", "type"],
                required_properties=["id", "location"],
                description="Storage facility in the supply chain",
                justified_by="Inventory management requires warehouse nodes for stock tracking",
            ),
            NodeSchema(
                label="Shipment",
                properties=["id", "shipped_at", "arrived_at", "carrier", "status"],
                required_properties=["id", "shipped_at", "status"],
                description="Physical movement of goods between locations",
                justified_by="Delivery tracking and lead time analysis",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="SUPPLIES",
                from_label="Supplier",
                to_label="Product",
                properties=["contract_start", "unit_price"],
                description="Supplier provides a product or component",
                direction_rationale="Supplier is the source, product is what they provide",
                justified_by="Multi-supplier risk analysis requires supplier→product edges",
            ),
            RelationshipSchema(
                type="STORED_IN",
                from_label="Product",
                to_label="Warehouse",
                properties=["quantity", "last_restocked"],
                description="Product inventory at a warehouse",
                direction_rationale="Product is stored in a warehouse location",
                justified_by="Inventory queries need product→warehouse stock levels",
            ),
            RelationshipSchema(
                type="SHIPPED_VIA",
                from_label="Product",
                to_label="Shipment",
                properties=["quantity"],
                description="Product included in a shipment",
                direction_rationale="Product is the thing being shipped",
                justified_by="Shipment tracking requires product→shipment edges",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT supplier_id IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT product_sku IF NOT EXISTS FOR (p:Product) REQUIRE p.sku IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX shipment_status IF NOT EXISTS FOR (sh:Shipment) ON (sh.status)",
            "CREATE INDEX warehouse_location IF NOT EXISTS FOR (w:Warehouse) ON (w.location)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT supplier_id IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE;\n"
            "CREATE CONSTRAINT product_sku IF NOT EXISTS FOR (p:Product) REQUIRE p.sku IS UNIQUE;\n"
            "CREATE INDEX shipment_status IF NOT EXISTS FOR (sh:Shipment) ON (sh.status);\n"
            "CREATE INDEX warehouse_location IF NOT EXISTS FOR (w:Warehouse) ON (w.location);"
        ),
    )


def social_network_schema() -> GraphSchema:
    """Social network — from Neo4j's classic movie/social graph tutorial.

    Source: Neo4j Developer guides, "Build a social network" tutorial.
    """
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Person",
                properties=["id", "name", "email", "joined_at", "bio"],
                required_properties=["id", "name", "email"],
                description="A user in the social network",
                justified_by="People are the core entity in any social graph",
            ),
            NodeSchema(
                label="Post",
                properties=["id", "content", "created_at", "visibility"],
                required_properties=["id", "content", "created_at"],
                description="User-generated content",
                justified_by="Content feeds require post nodes for timeline queries",
            ),
            NodeSchema(
                label="Community",
                properties=["id", "name", "description", "created_at"],
                required_properties=["id", "name"],
                description="Interest group or community",
                justified_by="Community detection and recommendation algorithms",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="FOLLOWS",
                from_label="Person",
                to_label="Person",
                properties=["since"],
                description="One user follows another",
                direction_rationale="Follow is directional — A follows B doesn't mean B follows A",
                justified_by="Feed algorithm needs follower graph traversal",
            ),
            RelationshipSchema(
                type="AUTHORED",
                from_label="Person",
                to_label="Post",
                properties=[],
                description="Person created a post",
                direction_rationale="Person is the creator, post is the artifact",
                justified_by="Content attribution and user profile queries",
            ),
            RelationshipSchema(
                type="MEMBER_OF",
                from_label="Person",
                to_label="Community",
                properties=["role", "joined_at"],
                description="Person belongs to a community",
                direction_rationale="Person joins community, not the reverse",
                justified_by="Community member listing and recommendation",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT person_email IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_at)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT person_id IF NOT EXISTS "
            "FOR (p:Person) REQUIRE p.id IS UNIQUE;\n"
            "CREATE CONSTRAINT person_email IF NOT EXISTS "
            "FOR (p:Person) REQUIRE p.email IS UNIQUE;\n"
            "CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_at);"
        ),
    )


def fraud_detection_schema() -> GraphSchema:
    """Fraud detection — from Neo4j's fraud detection whitepaper.

    Source: Neo4j "Fraud Detection" use case guide and GraphConnect talks.
    """
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Customer",
                properties=["id", "ssn_hash", "name", "dob", "risk_score"],
                required_properties=["id", "ssn_hash"],
                description="Bank customer subject to KYC checks",
                justified_by="KYC/AML regulations require unique customer identification",
            ),
            NodeSchema(
                label="Account",
                properties=["id", "iban", "type", "opened_at", "balance"],
                required_properties=["id", "iban", "type"],
                description="Financial account linked to a customer",
                justified_by="Transaction monitoring requires account-level tracking",
            ),
            NodeSchema(
                label="Transaction",
                properties=["id", "amount", "currency", "timestamp", "channel"],
                required_properties=["id", "amount", "timestamp"],
                description="Financial transaction between accounts",
                justified_by="Fraud ring detection requires transaction-level granularity",
            ),
            NodeSchema(
                label="Device",
                properties=["id", "fingerprint", "ip_address", "user_agent"],
                required_properties=["id", "fingerprint"],
                description="Device used to initiate transactions",
                justified_by="Device sharing between accounts is a key fraud signal",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="OWNS",
                from_label="Customer",
                to_label="Account",
                properties=["since"],
                description="Customer owns an account",
                direction_rationale="Customer is the owner, account is the asset",
                justified_by="Account ownership graph for KYC compliance",
            ),
            RelationshipSchema(
                type="SENT",
                from_label="Account",
                to_label="Transaction",
                properties=[],
                description="Account initiated a transaction",
                direction_rationale="Account is the sender, transaction is the event",
                justified_by="Fraud ring detection via sender path analysis",
            ),
            RelationshipSchema(
                type="RECEIVED",
                from_label="Transaction",
                to_label="Account",
                properties=[],
                description="Transaction received by account",
                direction_rationale="Money flows from transaction to receiving account",
                justified_by="Receiver pattern analysis for mule account detection",
            ),
            RelationshipSchema(
                type="USED_DEVICE",
                from_label="Account",
                to_label="Device",
                properties=["last_seen"],
                description="Account accessed from a device",
                direction_rationale="Account uses device, not the reverse",
                justified_by="Device sharing detection for fraud ring identification",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT account_iban IF NOT EXISTS FOR (a:Account) REQUIRE a.iban IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX device_fp IF NOT EXISTS FOR (d:Device) ON (d.fingerprint)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT customer_id IF NOT EXISTS "
            "FOR (c:Customer) REQUIRE c.id IS UNIQUE;\n"
            "CREATE CONSTRAINT account_iban IF NOT EXISTS "
            "FOR (a:Account) REQUIRE a.iban IS UNIQUE;\n"
            "CREATE INDEX tx_timestamp IF NOT EXISTS "
            "FOR (t:Transaction) ON (t.timestamp);\n"
            "CREATE INDEX device_fp IF NOT EXISTS FOR (d:Device) ON (d.fingerprint);"
        ),
    )


# ---------------------------------------------------------------------------
# KNOWN BAD — plausible but structurally broken schemas
# ---------------------------------------------------------------------------


def circular_only_schema() -> GraphSchema:
    """All relationships are self-referential loops. Structurally questionable."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Node",
                properties=["id"],
                required_properties=["id"],
                description="A generic node",
                justified_by="needed",
            ),
            NodeSchema(
                label="Edge",
                properties=["id"],
                required_properties=["id"],
                description="An edge pretending to be a node",
                justified_by="needed",
            ),
            NodeSchema(
                label="Link",
                properties=["id"],
                required_properties=["id"],
                description="Another generic name",
                justified_by="needed",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="CONNECTS_TO",
                from_label="Node",
                to_label="Node",
                properties=[],
                description="Self loop",
                direction_rationale="",
                justified_by="",
            ),
            RelationshipSchema(
                type="LINKS_TO",
                from_label="Edge",
                to_label="Edge",
                properties=[],
                description="Self loop",
                direction_rationale="",
                justified_by="",
            ),
        ],
        constraints=[],
        indexes=[],
        cypher_setup="// no setup",
    )


def dangling_references_schema() -> GraphSchema:
    """Relationships reference labels that don't exist in nodes."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="User",
                properties=["id", "name"],
                required_properties=["id"],
                description="A user",
                justified_by="User management system requires user entity",
            ),
            NodeSchema(
                label="Order",
                properties=["id", "total"],
                required_properties=["id"],
                description="A purchase order",
                justified_by="E-commerce order tracking",
            ),
            NodeSchema(
                label="Product",
                properties=["id", "name"],
                required_properties=["id"],
                description="A product",
                justified_by="Product catalog",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="PLACED",
                from_label="User",
                to_label="Order",
                properties=[],
                description="User placed an order",
                direction_rationale="User initiates order",
                justified_by="Order attribution to customer",
            ),
            RelationshipSchema(
                type="SHIPPED_TO",
                from_label="Order",
                to_label="Address",  # Address doesn't exist in nodes!
                properties=[],
                description="Order shipped to address",
                direction_rationale="Order goes to address",
                justified_by="Shipping logistics requires address endpoint",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX order_total IF NOT EXISTS FOR (o:Order) ON (o.total)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;\n"
            "CREATE INDEX order_total IF NOT EXISTS FOR (o:Order) ON (o.total);"
        ),
    )


def properties_mismatch_schema() -> GraphSchema:
    """Required properties not in properties list — consistency error."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Patient",
                properties=["id", "name"],
                required_properties=["id", "name", "ssn"],  # ssn not in properties!
                description="Hospital patient",
                justified_by="HIPAA requires patient identification for audit trails",
            ),
            NodeSchema(
                label="Doctor",
                properties=["id", "name", "specialty"],
                required_properties=["id", "license_number"],  # license_number not in properties!
                description="Treating physician",
                justified_by="Medical licensing requires physician tracking",
            ),
            NodeSchema(
                label="Diagnosis",
                properties=["id", "icd_code", "date"],
                required_properties=["id", "icd_code"],
                description="Medical diagnosis",
                justified_by="Clinical decision support requires diagnosis records",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="TREATED_BY",
                from_label="Patient",
                to_label="Doctor",
                properties=["date"],
                description="Patient treated by doctor",
                direction_rationale="Patient receives treatment from doctor",
                justified_by="Treatment history for patient outcomes analysis",
            ),
            RelationshipSchema(
                type="DIAGNOSED_WITH",
                from_label="Patient",
                to_label="Diagnosis",
                properties=["date"],
                description="Patient received diagnosis",
                direction_rationale="Patient is diagnosed, diagnosis is the finding",
                justified_by="Diagnostic history for clinical decision support",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX diagnosis_icd IF NOT EXISTS FOR (d:Diagnosis) ON (d.icd_code)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE;\n"
            "CREATE INDEX diagnosis_icd IF NOT EXISTS FOR (d:Diagnosis) ON (d.icd_code);"
        ),
    )


# ---------------------------------------------------------------------------
# ADVERSARIAL — designed to game the validator
# ---------------------------------------------------------------------------


def checkbox_stuffer_schema() -> GraphSchema:
    """Every field is filled in perfectly, but the schema is nonsense.

    All structural checks pass: 3+ nodes, 2+ relationships, constraints,
    indexes, justifications >20 chars. But the domain makes no sense —
    "Banana" connected to "Gravity" via "TRANSCENDS".
    """
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Banana",
                properties=["id", "color", "ripeness"],
                required_properties=["id", "color"],
                description="A tropical fruit in the schema",
                justified_by="Banana entities are required for fruit supply chain compliance",
            ),
            NodeSchema(
                label="Gravity",
                properties=["id", "force", "direction"],
                required_properties=["id", "force"],
                description="A fundamental force of nature",
                justified_by="Gravity modeling needed for physics simulation requirements",
            ),
            NodeSchema(
                label="TaxReturn",
                properties=["id", "year", "amount", "status"],
                required_properties=["id", "year"],
                description="Annual tax filing document",
                justified_by="Tax compliance requires annual return tracking per entity",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="TRANSCENDS",
                from_label="Banana",
                to_label="Gravity",
                properties=["intensity"],
                description="Banana transcends gravitational force",
                direction_rationale="Banana is the actor transcending gravity's pull",
                justified_by="Transcendence modeling for fruit-gravity interaction analysis",
            ),
            RelationshipSchema(
                type="AUDITS",
                from_label="Gravity",
                to_label="TaxReturn",
                properties=["date"],
                description="Gravity audits the tax return",
                direction_rationale="Gravity is the auditing entity for tax compliance",
                justified_by="Gravitational auditing process for tax return verification",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT banana_id IF NOT EXISTS FOR (b:Banana) REQUIRE b.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX tax_year IF NOT EXISTS FOR (t:TaxReturn) ON (t.year)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT banana_id IF NOT EXISTS FOR (b:Banana) REQUIRE b.id IS UNIQUE;\n"
            "CREATE INDEX tax_year IF NOT EXISTS FOR (t:TaxReturn) ON (t.year);"
        ),
    )


def keyword_stuffer_schema() -> GraphSchema:
    """Justifications are keyword-stuffed to game fulltext search.

    Every justification contains "Neo4j", "best practice", "graph modeling",
    "performance", etc. but says nothing specific.
    """
    filler = (
        "Neo4j graph database best practice pattern for optimal "
        "performance and scalability in enterprise knowledge graph modeling"
    )
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Alpha",
                properties=["id", "value"],
                required_properties=["id"],
                description="Generic alpha node",
                justified_by=f"Alpha required for {filler}",
            ),
            NodeSchema(
                label="Beta",
                properties=["id", "value"],
                required_properties=["id"],
                description="Generic beta node",
                justified_by=f"Beta required for {filler}",
            ),
            NodeSchema(
                label="Gamma",
                properties=["id", "value"],
                required_properties=["id"],
                description="Generic gamma node",
                justified_by=f"Gamma required for {filler}",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="LINKS",
                from_label="Alpha",
                to_label="Beta",
                properties=[],
                description="Alpha links to Beta",
                direction_rationale=f"Direction follows {filler}",
                justified_by=f"Link pattern per {filler}",
            ),
            RelationshipSchema(
                type="CHAINS",
                from_label="Beta",
                to_label="Gamma",
                properties=[],
                description="Beta chains to Gamma",
                direction_rationale=f"Chain direction per {filler}",
                justified_by=f"Chaining pattern per {filler}",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT alpha_id IF NOT EXISTS FOR (a:Alpha) REQUIRE a.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX beta_value IF NOT EXISTS FOR (b:Beta) ON (b.value)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT alpha_id IF NOT EXISTS FOR (a:Alpha) REQUIRE a.id IS UNIQUE;\n"
            "CREATE INDEX beta_value IF NOT EXISTS FOR (b:Beta) ON (b.value);"
        ),
    )


def structural_pass_semantic_fail_schema() -> GraphSchema:
    """Passes all structural checks perfectly but is semantically absurd.

    A healthcare schema where "Surgery" EMPLOYS "Medication" and
    "Medication" DIAGNOSES "Hospital". Structure is perfect, meaning is garbage.
    """
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Surgery",
                properties=["id", "type", "duration", "risk_level"],
                required_properties=["id", "type", "risk_level"],
                description="A surgical procedure performed on a patient",
                justified_by="Surgical procedures require tracking for clinical outcomes analysis",
            ),
            NodeSchema(
                label="Medication",
                properties=["id", "name", "dosage", "manufacturer"],
                required_properties=["id", "name", "dosage"],
                description="A pharmaceutical medication",
                justified_by="Medication tracking required for adverse reaction monitoring",
            ),
            NodeSchema(
                label="Hospital",
                properties=["id", "name", "capacity", "accreditation"],
                required_properties=["id", "name"],
                description="A healthcare facility",
                justified_by="Hospital tracking for regional healthcare capacity planning",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="EMPLOYS",
                from_label="Surgery",
                to_label="Medication",
                properties=["date"],
                description="Surgery employs medication",
                direction_rationale="Surgery is the employer, medication is the employee",
                justified_by="Employment tracking for surgical-pharmaceutical analysis",
            ),
            RelationshipSchema(
                type="DIAGNOSES",
                from_label="Medication",
                to_label="Hospital",
                properties=["confidence"],
                description="Medication diagnoses a hospital",
                direction_rationale="Medication is the diagnosing agent for hospital assessment",
                justified_by="Diagnostic capability modeling for medication-facility analytics",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT surgery_id IF NOT EXISTS FOR (s:Surgery) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT med_id IF NOT EXISTS FOR (m:Medication) REQUIRE m.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX hospital_name IF NOT EXISTS FOR (h:Hospital) ON (h.name)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT surgery_id IF NOT EXISTS FOR (s:Surgery) REQUIRE s.id IS UNIQUE;\n"
            "CREATE CONSTRAINT med_id IF NOT EXISTS FOR (m:Medication) REQUIRE m.id IS UNIQUE;\n"
            "CREATE INDEX hospital_name IF NOT EXISTS FOR (h:Hospital) ON (h.name);"
        ),
    )


# ---------------------------------------------------------------------------
# Convenience collections for parametrized tests
# ---------------------------------------------------------------------------

KNOWN_GOOD = [supply_chain_schema, social_network_schema, fraud_detection_schema]
KNOWN_BAD = [circular_only_schema, dangling_references_schema, properties_mismatch_schema]
ADVERSARIAL = [
    checkbox_stuffer_schema,
    keyword_stuffer_schema,
    structural_pass_semantic_fail_schema,
]
