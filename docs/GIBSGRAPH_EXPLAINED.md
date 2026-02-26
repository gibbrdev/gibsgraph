# GibsGraph — The Complete Guide

> Ask your business data anything. In plain English. Get answers you can trust.

---

## What Is GibsGraph?

Imagine you could talk to your company's data like you talk to a colleague.

Not "run this SQL query" or "build me a dashboard." Just:

- *"Which suppliers are single points of failure?"*
- *"Show me everyone connected to this fraudulent transaction."*
- *"Are we GDPR compliant in the payment flow?"*

And you get a real answer — with the exact data that proves it, the sources it came from, and a visual map of the connections.

That's GibsGraph.

It sits on top of Neo4j (the world's most popular graph database) and turns it from something only developers can use into something anyone in your organization can query. No code. No training. No waiting for IT.

---

## Why Graphs? (30 seconds)

Your data lives in tables — spreadsheets, databases, CRMs, ERPs. Tables are great for lists. But your business doesn't run on lists. It runs on **connections**.

- A customer **owns** an account that **sent** a transaction to a **merchant** using a **device** that's shared with **another account** — that's a fraud ring.
- A supplier **provides** a component that **goes into** a product that **ships to** a warehouse that **serves** a region — that's a supply chain vulnerability.
- A regulation **requires** a control that **applies to** a process that **handles** personal data — that's a compliance gap.

Tables can't show you these connections. Graphs can. And GibsGraph lets you explore them by just asking questions.

---

## Real Use Cases

### Financial Services: Fraud Detection

**The situation:** Your fraud team manually reviews flagged transactions. They catch obvious cases but miss sophisticated fraud rings where multiple accounts share devices, addresses, or beneficiaries.

**With GibsGraph:**
```
"Show me accounts that share a device with the flagged account A-4829"
```
GibsGraph traverses the graph: Account → Device → Other Accounts → Their Transactions. It returns a map showing 6 connected accounts, 3 shared devices, and 47 transactions between them. Your fraud analyst sees the full ring in seconds, not days.

**Other questions you could ask:**
- *"Which customers have sent money to more than 5 new beneficiaries this week?"*
- *"Find all accounts within 3 hops of this sanctioned entity"*
- *"What's the average transaction amount for this merchant compared to similar merchants?"*

---

### Healthcare: Patient Safety & Compliance

**The situation:** Your hospital tracks patients, doctors, medications, and diagnoses across separate systems. When a drug interaction risk appears, no one has a unified view.

**With GibsGraph:**
```
"Which patients are currently prescribed medications that interact with each other?"
```
GibsGraph connects Patient → Prescription → Medication → Known Interactions → Other Medications → Same Patient. It flags 12 patients with potential conflicts, sorted by severity.

**Other questions:**
- *"Show me Dr. Smith's patient outcomes compared to department average"*
- *"Which departments have the longest time between diagnosis and treatment?"*
- *"Are all HIPAA-required audit trails in place for patient transfers?"*

---

### Supply Chain: Risk & Resilience

**The situation:** You have 200 suppliers, but you don't know which ones are irreplaceable. When one factory in Taiwan shut down, it took 3 weeks to figure out which products were affected.

**With GibsGraph:**
```
"Which products depend on a single supplier with no backup?"
```
GibsGraph maps: Product → Component → Supplier. It identifies 14 products with single-source dependencies, ranked by revenue impact. You now have a prioritized list for supplier diversification.

**Other questions:**
- *"If Supplier X shuts down, which warehouses are affected within 30 days?"*
- *"Show me the full journey of Product SKU-8812 from raw material to customer"*
- *"Which shipping routes have the highest delay rates this quarter?"*

---

### Compliance & Regulatory: Audit Readiness

**The situation:** Your compliance team spends 2 months preparing for audits. They manually trace which regulations apply to which processes, which controls are in place, and where the gaps are.

**With GibsGraph:**
```
"Show me all GDPR requirements that don't have a documented control"
```
GibsGraph traverses: Regulation → Requirement → Control (missing). It returns 8 unaddressed requirements with their risk level and the processes they affect. Audit prep goes from 2 months to 2 hours.

**Other questions:**
- *"Which business processes handle personal data without encryption?"*
- *"Map all SOX controls to their evidence documents — what's missing?"*
- *"If we expand to the EU market, which additional regulations apply to our current products?"*

---

### E-Commerce: Customer Intelligence

**The situation:** Your marketing team wants to understand customer behavior beyond "bought X." They want to know the story — what led to the purchase, what they browsed, who influenced them, what they'll want next.

**With GibsGraph:**
```
"Show me the purchase journey for customers who bought Product X but returned it"
```
GibsGraph maps: Customer → Browsed → Product Pages → Added to Cart → Purchased → Returned. It reveals that 73% of returns came from customers who never viewed the size guide, and most were influenced by a specific ad campaign.

**Other questions:**
- *"Which products are frequently bought together but never recommended together?"*
- *"Find customers who are one purchase away from loyalty tier upgrade"*
- *"Which influencer's referral customers have the highest lifetime value?"*

---

### Cybersecurity: Threat Detection

**The situation:** Your SOC team sees thousands of alerts daily. Most are noise. The real attacks span multiple systems, users, and time windows — and they're invisible in individual log entries.

**With GibsGraph:**
```
"Show me all lateral movement paths from the compromised endpoint to sensitive data"
```
GibsGraph traces: Compromised Machine → Credentials Used → Other Machines Accessed → Network Segments → Data Stores Reached. It shows the attacker could reach your customer database in 4 hops through a forgotten service account.

**Other questions:**
- *"Which users have access to both production databases and external email?"*
- *"Map all authentication paths to our crown jewel systems"*
- *"If this VPN credential is compromised, what's the blast radius?"*

---

## How It Works — The Simple Version

```
┌─────────────────────────────────────────────┐
│                                             │
│   You: "Which suppliers are at risk?"       │
│                                             │
│              ↓                              │
│                                             │
│   GibsGraph understands your question       │
│   Searches your data graph                  │
│   Finds the relevant connections            │
│   Checks its work for safety                │
│   Generates a clear answer                  │
│                                             │
│              ↓                              │
│                                             │
│   Answer: "3 suppliers are single-source    │
│   for high-revenue products..."             │
│   + visual map + source citations           │
│                                             │
└─────────────────────────────────────────────┘
```

Behind the scenes, there's a 4-step pipeline that runs in seconds:

**Step 1: Understand** — GibsGraph figures out what you're really asking. "Which suppliers are at risk?" becomes a structured search for single-source dependencies with no backup suppliers.

**Step 2: Search** — It automatically picks the best search strategy for your question. For simple lookups, it goes straight to the data. For complex questions, it writes a database query, checks it for safety, and runs it.

**Step 3: Verify** — Every query is checked for safety before it touches your data. No deletions, no modifications, no injections. Read-only by default.

**Step 4: Explain** — The raw data gets turned into a clear answer with a visual graph map and a direct link to explore further in Neo4j Bloom.

---

## How It Works — The Full Picture

For those who want to understand every piece.

### The Entry Point: Graph

This is the only thing you interact with. One class. Four methods.

**`Graph()`** — Connect to your database.
Set up your connection once. Provide your Neo4j address and password. Everything else is automatic — it detects which AI model you have available (OpenAI, Anthropic, Mistral, or Grok) and configures itself.

**`g.ask("your question")`** — Ask anything in plain English.
Returns an answer with:
- The answer itself (plain text you can read)
- The database query it used (for transparency)
- A confidence score (how sure it is)
- A visual map of the data it found
- A link to explore the results interactively
- Any errors or caveats

**`g.ingest("your text")`** — Feed it new data. *(Coming soon)*
Give it documents, and it builds the knowledge graph for you. Upload contracts, reports, CSVs — GibsGraph extracts the entities and relationships automatically.

**`g.visualize("your question")`** — Get a visual map.
Returns a diagram showing the nodes and connections related to your question. Useful for presentations, reports, or just understanding complex relationships at a glance.

---

### The Brain: The Agent Pipeline

When you ask a question, it flows through four stages — like an assembly line where each station does one specific job.

**Stage 1 — Retrieve**
The retriever looks at your question and decides how to search your graph. It has two strategies:

- *Vector search* — If your graph has semantic embeddings (think: "meaning-aware search"), it finds nodes that are conceptually similar to your question, then expands outward to grab connected data. Best for exploratory questions like "show me everything related to X."

- *Cypher generation* — For precise questions, it writes a database query in Neo4j's query language (Cypher). It uses AI to translate your English question into a precise graph query. If the first attempt fails, it learns from the error and tries again.

Both strategies automatically discover your graph's structure first — what types of data you have, how they're connected, what properties exist. You never have to tell it about your schema.

**Stage 2 — Explain**
The raw data from the search gets sent to an AI model that generates a clear, natural language answer. It doesn't just dump database rows — it synthesizes the findings into something a human can read and act on.

**Stage 3 — Validate**
Before anything leaves the system, every database query is checked for safety:

- No deletions or modifications (read-only)
- No injection attacks (parameterized queries only)
- No dangerous operations (no DROP, DELETE, REMOVE)
- No access to system procedures

If a query fails validation, it gets flagged for manual review instead of executing.

**Stage 4 — Visualize**
The results are converted into a visual graph map (Mermaid diagram) and a direct link to Neo4j Bloom for interactive exploration. You can share these with colleagues who want to dig deeper.

---

### The Expert: The Knowledge Graph

This is what makes GibsGraph different from every other "AI on a database" tool.

Inside GibsGraph lives an **expert knowledge graph** — a curated database of Neo4j expertise:

- **36 Cypher clauses** — every command in the Neo4j query language, with descriptions and usage examples
- **122 built-in functions** — every function Neo4j provides, with signatures and return types
- **383 real code examples** — working Cypher queries from official documentation
- **20 modeling patterns** — expert knowledge about when to use nodes vs. properties, how to model hierarchies, time-series patterns
- **309 best practices** — operational wisdom about indexing, constraints, query optimization, security

When GibsGraph writes a database query to answer your question, it consults this expert knowledge. It's like having a senior Neo4j consultant looking over every query before it runs.

**Why this matters to you:** The queries GibsGraph writes aren't generic AI guesses. They're informed by the same patterns and best practices that Neo4j's own engineers recommend. This means faster queries, fewer errors, and more accurate results.

---

### The Quality Gate: Validation Suite

When GibsGraph generates a new graph schema (the blueprint for how your data should be organized), it goes through a 4-stage quality check before it's approved:

**Stage 1 — Syntactic Check: "Is it safe?"**
Checks that the database setup script is safe to run. No destructive operations, no dangerous keywords, no references to things that don't exist. Think of it as a safety inspector checking the blueprints before construction begins.

**Stage 2 — Structural Check: "Is it complete?"**
Verifies that the schema has all the required pieces:
- Enough node types to represent the domain (minimum 3)
- Relationships between them (minimum 2)
- Database constraints to enforce data integrity
- Indexes for query performance
- Every design decision justified by a real requirement

This check was specifically built to reject empty or half-finished schemas. A blank schema scores near zero, not a passing grade.

**Stage 3 — Semantic Check: "Is it real?"**
Queries the actual live database to verify the schema matches reality:
- Do the labels actually exist in the database?
- Do the relationship types exist?
- Are required fields actually populated (not null)?
- Are there orphan nodes disconnected from everything else?

This stage requires a live database connection. No connection = no score. We don't give credit for claims we can't verify.

**Stage 4 — Domain Check: "Is it good?"**
Evaluates the quality of the Cypher setup script. Does it create proper constraints? Does it set up the right indexes? Are there any security concerns?

**The approval logic:** A schema needs to score above 70% overall AND have zero critical errors. A warning (like a missing index) degrades the score but doesn't block approval. An error (like a reference to a non-existent label) blocks approval regardless of score.

**Why this matters to you:** Every graph GibsGraph builds or validates has been through this quality gate. You're not trusting blind AI output — you're trusting output that's been checked against real data, real patterns, and real best practices.

---

### The Safety Layer: Cypher Validator

Every database query that GibsGraph generates is checked for safety before execution. This is non-negotiable — it runs on every single query, every single time.

**What it catches:**
- **Destructive operations** — CREATE, DELETE, DROP, DETACH, REMOVE. GibsGraph is read-only by default. It asks questions, it doesn't change your data.
- **Injection attacks** — If someone tries to sneak malicious code into a question, the validator catches it. All queries use parameterized inputs (the gold standard for database security).
- **Dangerous procedures** — Calls to system procedures, bulk operations, CSV loading — all blocked.
- **String interpolation** — No f-strings, no string concatenation in queries. This is the #1 source of database security vulnerabilities, and GibsGraph blocks it completely.

**Why this matters to you:** Your data is safe. GibsGraph physically cannot delete, modify, or corrupt your data. The safety checks aren't optional — they're built into the architecture. Even if the AI model tries to generate a dangerous query, it gets caught before it ever touches your database.

---

### The Visualizer

Every answer comes with visuals — because sometimes a picture really is worth a thousand words.

**Graph Maps (Mermaid diagrams)** — A flowchart showing the nodes and relationships in your answer. These can be embedded in documents, presentations, or dashboards. They're generated automatically from the query results.

**Interactive Explorer (Neo4j Bloom)** — A direct link that opens your results in Neo4j's interactive visualization tool. You can click on nodes, expand connections, filter by properties — full exploration without writing any queries.

**Network Diagrams (PyVis)** — Interactive HTML network visualizations you can embed in web pages or share with colleagues. Nodes are clickable, draggable, and color-coded by type.

---

### Configuration: One File, Everything Set

GibsGraph configures itself from environment variables or a `.env` file. You set it once and forget it.

**Database connection:**
- Where your Neo4j database lives (address, username, password)
- Which database to use (if you have multiple)
- Read-only mode (on by default — your data is safe)

**AI model:**
- Which AI provider to use (OpenAI, Anthropic, Mistral, or Grok)
- GibsGraph auto-detects which one you have based on your API keys
- You can specify a model or let it pick the best available one

**Security:**
- Rate limiting (60 queries per minute by default)
- Connection timeouts
- All passwords stored securely (never logged, never exposed)

**Observability:**
- Optional LangSmith integration for tracing queries
- Structured logging for every operation

---

### The Command Line

For quick access without writing any code:

```
gibsgraph ask "Which customers have the highest risk score?"
```

That's it. One command. The answer prints to your terminal with the Cypher query used, a visual map, and a link to explore further.

```
gibsgraph ingest contracts/supplier-agreement.pdf
```

Feed it a document, and it builds the knowledge graph automatically. *(Coming soon)*

---

## What's Built Today vs. What's Coming

| Feature | Status | What It Does |
|---------|--------|-------------|
| `g.ask()` | Live | Ask questions in plain English, get answers with sources |
| Expert Knowledge Graph | Live | 36 clauses, 122 functions, 383 examples, 309 best practices |
| 4-Stage Validation | Live | Every schema checked: syntax, structure, data quality, domain |
| Auto Schema Discovery | Live | GibsGraph learns your graph structure automatically |
| Cypher Safety Validator | Live | Every query checked for safety before execution |
| Vector + Cypher Retrieval | Live | Two search strategies, auto-selected per question |
| Visual Graph Maps | Live | Mermaid diagrams, Bloom links, interactive HTML |
| Multi-Model AI | Live | OpenAI, Anthropic, Mistral, Grok — auto-detected |
| CLI | Live | `gibsgraph ask` from your terminal |
| `g.ingest()` | Coming | Documents to knowledge graph, automatically |
| GNN Training | Coming | The system gets smarter with every graph it builds |
| `g.map()` | Coming | Describe your business, get a production graph |

---

## The Trust Architecture

GibsGraph was built by someone who doesn't trust AI output. Every layer has a check.

**Query safety** — Every database query validated before execution. No exceptions.

**Read-only by default** — GibsGraph cannot modify your data unless you explicitly allow it.

**No benefit-of-the-doubt scoring** — If the system can't verify a claim against real data, the score is 0. Not 0.5. Not "probably fine." Zero.

**Anti-gaming protection** — The validation suite was tested with adversarial schemas specifically designed to cheat the system. Nonsense schemas with all the right buzzwords? Blocked. Keyword-stuffed justifications? Blocked. Structurally perfect but semantically absurd? Blocked.

**Independent test data** — The test schemas used to verify the validator were written without knowledge of how the validator works. We don't grade our own homework.

**Source citations** — Every answer traces back to the actual data that produced it. You can verify any claim by following the trail.

---

## One Last Thing

GibsGraph is open source. The code is public. The expert knowledge graph is public. The validation suite is public.

You can read every line. You can audit the safety checks. You can verify the expert patterns. You can run the quality tests yourself.

We believe the best way to earn trust is to be completely transparent about how the system works. No black boxes. No "trust us." Just open code and honest scoring.

**Three lines. That's all it takes.**

```python
from gibsgraph import Graph

g = Graph("bolt://localhost:7687", password="your-password")
answer = g.ask("What do you see in my data?")
```

---

*GibsGraph is built and maintained by [Victor Gibson](https://gibs.dev). Open source under MIT license.*
*Repository: [github.com/gibbrdev/gibsgraph](https://github.com/gibbrdev/gibsgraph)*
