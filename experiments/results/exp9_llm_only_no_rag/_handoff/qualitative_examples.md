# Qualitative examples — exp9_llm_only_no_rag

Ten illustrative responses extracted from the saved `results.json`. All texts are verbatim from Llama 3.1 8B Q4 with `NO_RAG_SYSTEM_PROMPT`. Verification of factual claims (where performed) uses official vendor documentation accessed 2026-05-21.

---

## FABRICATION FLAGRANT (3)

### q001 — "Limits and quotas for AWS EKS"

> According to my knowledge, AWS EKS (Elastic Kubernetes Service) has several limits and quotas that apply to clusters, nodes, and other resources. Here are some of them:
>
> **Cluster Limits**
>
> * Maximum number of clusters per account: 100
> * Maximum number of worker node groups per cluster: 50
> * Maximum number of worker nodes per group: 100
>
> **Node Limits**
>
> * Maximum number of nodes per cluster: 1000 (for standard clusters) or 2000 (for high availability clusters)
> * Maximum number of node groups per cluster: 50
> * Minimum and maximum number of nodes in a group: 1-100

**Fabrication evidence**: EKS does have soft quotas, but the LLM fabricates the structural categorisation. Real EKS quotas are per-region per-account (default 100 clusters per region), not per-account. The invented "standard clusters" / "high availability clusters" split (1000 vs 2000 nodes) does not exist as a documented limit class in EKS — node-count limits are determined by instance type and pod density, not a binary cluster-class flag. Hallucination is flagrant because the structure itself is invented.

### q002 — "Pricing tiers for AWS EC2"

> According to my knowledge, AWS offers several pricing tiers for Amazon Elastic Compute Cloud (EC2) instances:
>
> 1. **On-Demand Instances**: Pay as you go, with no upfront costs or long-term commitments.
> 2. **Reserved Instances**:
>     * **Lightly Used Reserved Instances**: For 25% utilization, with a one-year or three-year term.
>     * **Standard Reserved Instances**: For 50% utilization, with a one-year or three-year term.
>     * **Heavy Utilization Reserved Instances**: For 75% utilization, with a one-year or three-year term.

**Fabrication evidence**: AWS retired the Light/Medium/Heavy Utilization classification for Reserved Instances in late 2014 (~12 years ago) and replaced it with "Standard" and "Convertible" RIs (with optional Scheduled RIs). Quoting "Lightly Used Reserved Instances: For 25% utilization" as a current pricing tier is fabrication by the LLM. The 25%/50%/75% utilization thresholds attributed to those tiers also never existed in any AWS pricing model.

### q050 — "Available SKUs for Azure Entra ID"

> According to my knowledge, Azure Entra (formerly known as Azure Active Directory) offers several pricing tiers and SKUs to suit different needs and requirements. Here are some of the available SKUs:
>
> 1. **Azure AD Free**: This is a free tier that provides basic identity and access management capabilities.
> 2. **Azure AD Premium P1** (also known as Azure AD Premium): This SKU adds advanced features such as multi-factor authentication, self-service password reset, and group-based licensing.
> 3. **Azure AD Premium P2**: This SKU includes all the features of P1, plus additional advanced security features like privileged identity management and cloud app security.
> 4. **Azure AD Premium P3** (also known as Azure AD Premium with MFA): This SKU is specifically designed for organizations that require [...]

**Fabrication evidence** (verified via [https://www.microsoft.com/en-us/security/business/microsoft-entra-pricing](https://www.microsoft.com/en-us/security/business/microsoft-entra-pricing), accessed 2026-05-21): Microsoft Entra ID has Free, P1 ($6/user/mo), P2 ($9/user/mo), and the Microsoft Entra Suite bundle ($12/user/mo). **No "P3" tier exists.** The LLM invented "Azure AD Premium P3 (also known as Azure AD Premium with MFA)" — both the name and the alias are fabricated. MFA is not a separate SKU; it is a feature included in P1 and above.

---

## FABRICATION SUBTLE (3) — specific numbers verifiable against official docs

### q003 — "SLA for AWS Lambda"

> According to my knowledge, AWS Lambda has a Service Level Agreement (SLA) that guarantees 99.99% execution availability within a region. This means that AWS Lambda will be available and functioning correctly at least 99.99% of the time in a given region.

**Fabrication evidence** (verified via [https://aws.amazon.com/lambda/sla/](https://aws.amazon.com/lambda/sla/), accessed 2026-05-21):

> "AWS will use commercially reasonable efforts to make Lambda available with a Monthly Uptime Percentage for each AWS region, during any monthly billing cycle, of at least **99.95%**."

The LLM's "99.99%" is **wrong by an order of magnitude in error budget** (99.99% allows ~4.3 min/month downtime; 99.95% allows ~21.6 min/month — 5× more). Additionally, the LLM's "execution availability" phrasing is non-standard; the AWS SLA targets "Monthly Uptime Percentage" measured per region. Subtle because the number reads as plausibly precise.

### q030 — "Azure Blob Storage capabilities"

> The main capabilities of Azure Blob Storage include:
>
> 1. **Unlimited storage**: Store an unlimited amount of data in the cloud.
> 2. **High scalability**: Easily scale up or down to meet changing storage needs.

**Fabrication evidence**: Azure Blob Storage is NOT unlimited. Documented limits (Microsoft docs) include a default storage account capacity of 5 PiB (extendable on request), 200 TB per block blob (4.77 TB before 2021), and a max of 50,000 blocks per block blob. Asserting "unlimited" is a subtle but verifiable error — the kind of claim a reader of a thesis would not flag without checking docs.

### q150 — "Compare pricing of AWS CloudWatch vs AWS IAM"

> **AWS CloudWatch**: [...] The pricing model is based on the number of metrics, log groups, and data points stored in CloudWatch. There are two main components to consider:
>
> 1. **Metrics**: You're charged per metric, with a minimum charge of 10 units (regardless of usage). Each unit represents one metric, and you can have up to 1000 metrics.
> 2. **Logs**: You're charged for the number of log groups and data points stored in CloudWatch Logs.

**Fabrication evidence**: CloudWatch pricing has no "minimum charge of 10 units" and no "1000 metrics" cap. Custom metrics are billed per metric per month (with the first 10 free), and there is no documented hard limit on the number of metrics — only a default soft quota per region (e.g. 5,000 alarms per region, 10 metric streams per region, etc.). The "unit" abstraction is invented. Subtle because pricing tables are inherently complex; a casual reader would not notice the fabricated structure.

---

## EXTRACTOR FAILURE (1)

### q196 — "Equivalent networking concepts across AWS, Azure, and GCP for VPCs, subnets, and security groups"

> Based on my knowledge, here are the equivalent networking concepts across AWS, Azure, and GCP:
>
> * **VPC (Virtual Private Cloud)**:
>     + AWS: Virtual Private Cloud (VPC)
>     + Azure: Virtual Network (VN)
>     + GCP: Virtual Network (VPC) or Google Compute Engine network
> * **Subnets**:
>     + AWS: Subnet
>     + Azure: Subnet
>     + GCP: Subnetwork
> * **Security Groups**:
>     + AWS: Security Group
>     + Azure: Network Security [...]

**Why the extractor returned []** (analysis of `HallucinationDetector._extract_claims` against this text):

1. **Opening filtered**: First sentence "Based on my knowledge, here are the equivalent networking concepts across AWS, Azure, and GCP" begins with `^based on` which is in `SKIP_PATTERNS` (`src/generation/hallucination_detector.py:49`). Skipped.
2. **Body is bullets, not sentences**: Each bullet line ends with `:` (the header bullets like `**VPC (Virtual Private Cloud)**:`) or with no terminator at all (the sub-bullets like `+ AWS: Virtual Private Cloud (VPC)`). The claim splitter uses `re.split(r'(?<=[.!?])\s+', ...)` (line 219) which only splits on `.!?` followed by whitespace. The whole bulleted block collapses into a few extremely long "sentences" with no internal split points.
3. **Bullet headers filtered**: After concatenation, the resulting "sentences" still end in `:` and get caught by the rule `if sent.endswith(":"): continue` (line 237).
4. **Closing filtered**: The response's trailing "Note that the actual terminology may vary..." matches `^note that` in `SKIP_PATTERNS`.

Net: `claims = []`. `_no_evidence_report` took the empty-claims branch (intended for honest declines) and returned **`faithfulness_score=1.0`, `hallucination_rate=0.0` vacuously**. The faithfulness=1.0 is misleading because the response contains substantive (and mostly correct) factual content — the extractor simply could not parse list format.

**Implication for the paper**: any aggregate over LLM-only that includes q196 inherits 0.005 of upward bias on `hall_faithfulness_mean` and 0.005 of downward bias on `hall_hallucination_rate_mean`. The corrected mean (Block 1) excludes q196 → `faithfulness_mean_corrected=0.0`.

---

## HONEST DECLINE (3, selected from 4 identified in Block 1)

### q081 — "How to configure networking for Azure Entra ID?"
Decline pattern matched: `r"i'm not familiar with"`

> I'm not familiar with "Azure Entra ID". I think it might be a new service or feature in Azure, but I couldn't find any information about it. Can you please provide more context or clarify what Azure Entra ID is? I'll do my best to help once I understand the question better.
>
> However, if you meant "Azure Active Directory (AAD) Identity Protection" or another related service, I can try to provide a [...]

The LLM correctly recognises "Azure Entra ID" as outside its knowledge cutoff (Microsoft rebranded Azure AD to Entra ID in 2023; Llama 3.1 has Dec 2023 cutoff but may not have been trained on the rebrand). Honest decline with offer to attempt a related question. Note: total_claims=3 (some non-decline claims about AAD made downstream of the disclaimer), so faithfulness=0.0 in the saved record — correctly so, because those downstream claims are still unsupported under the no-evidence definition.

### q097 — "How to scale Azure Monitor horizontally?"
Decline pattern matched: `r"i don't have.*information"`

> Azure Monitor is a service that collects and analyzes data from various sources in an Azure environment. It's designed to provide insights into performance, availability, and usage of Azure resources.
>
> Unfortunately, I don't have information on how to scale Azure Monitor itself horizontally (i.e., adding more instances or nodes). However, I can suggest some related concepts:
>
> * You can configure A[...]

Clean honest decline on the specific scaling question. Preceded by a brief (and generally correct) description of what Azure Monitor is. The decline is for the operational specifics, not the conceptual overview.

### q117 — "How to implement auto-scaling with Google Cloud Cloud Run?"
Decline pattern matched: `r"i don't have.*information"`

> Unfortunately, I don't have specific information on implementing auto-scaling for Google Cloud Cloud Run. While I'm familiar with Cloud Run's capabilities and features, my knowledge on its auto-scaling configuration is limited.
>
> However, I can suggest some general guidance on how to implement auto-scaling in Cloud Run:
>
> 1. **Cloud Run Scaling**: You can enable scaling for your service by going to [...]

Honest decline followed by general guidance. The general guidance still produces claims that are marked unsupported_no_evidence (total_claims=10), so faithfulness=0.0 in the saved record. The HONEST_DECLINE confidence flag was NOT set by `response_formatter` because the existing `DECLINE_PATTERNS` are case-sensitive on capital-I patterns (see audit_report.md M1 note) and "unfortunately, i don't have specific information" requires the user's proposed extended lowercased pattern set to match.

**Fourth decline (not shown above): q106** "How to configure IAM permissions for Google Cloud VPC?" — matches `r"i'm not familiar with"`.

---

## Summary

The qualitative picture confirms the quantitative finding: Llama 3.1 8B Q4 with a permissive "say if not sure" system prompt fabricates confidently in 195/200 queries, honestly declines in 4/200, and produces one extractor-failure edge case (q196). Fabrications span flagrant (invented tiers, structurally wrong) and subtle (specific numbers verifiably wrong against official docs). This is the absolute floor against which RAG systems should be compared.
