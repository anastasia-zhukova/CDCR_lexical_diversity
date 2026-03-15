# Cross-Document Coreference Resolution: Results Analysis

## Approach

### Architecture

The model is a **RoBERTa-large encoder** (355M params) with dual MLP projection heads, trained with **Multi-Similarity (MS) Loss** for metric learning. Each mention is encoded as a 2048-dimensional L2-normalized embedding by concatenating:

- A **CLS projection** (1024-dim): the [CLS] token passed through a 3-layer MLP, capturing document-level context.
- A **Mention span projection** (1024-dim): the sum of tokens between `[E]` and `[/E]` markers passed through a separate 3-layer MLP, capturing entity-specific signal.

At inference, agglomerative clustering (average linkage) groups mention embeddings using a cosine distance threshold.

### Training strategy: PK Batch Sampling with PD weighting

Training uses a structured **PK batch sampler** that constructs each batch with controlled negative difficulty:

- 60% of chains from the same subtopic (hard negatives: similar context, different entity)
- 20% from the same topic, different subtopic (medium negatives)
- 20% random chains (easy negatives: diversity)

Chain sampling is **weighted by phrasing diversity (PD)**: `weight = PD^tau * log(chain_size) / log(dataset_chain_count)`. The `tau` parameter controls how aggressively high-PD chains (like those in NewsWCL50) are prioritized during training. Difficulty-aware sampling via an EMA on per-chain loss keeps hard chains in the training pool longer.

### Training data

The model trains on a **unified cross-document coreference corpus (uCDCR)** comprising 12 datasets with 127,958 training mentions across 15,695 coreference chains. This multi-dataset setup is critical: the model must learn a single embedding space that works across diverse annotation schemes, entity types, and document styles.

---

## Experimental runs

Three training runs were conducted, varying the PD sampling emphasis (`tau`) and MS Loss hyperparameters:

| Run | tau | ms-alpha | ms-lambda | diff-alpha | Best avg pd_LEA F1 | Best threshold | NewsWCL50 val pd_LEA F1 |
|-----|-----|----------|-----------|------------|---------------------|----------------|-------------------------|
| 1 (baseline) | 1.0 | 2.0 | 0.5 | 0.1 | **0.4290** | 0.5 | 0.3012 |
| 2 (aggressive) | 2.0 | 4.0 | 0.3 | 0.3 | 0.3512 | 0.5 | 0.2582 |
| 3 (targeted) | 1.5 | 2.0 | 0.5 | 0.1 | 0.4178 | 0.4 | **0.3149** |

**Run 2** attempted to simultaneously increase PD emphasis, strengthen positive-pair gradients, tighten the similarity margin, and accelerate difficulty annealing. This destabilized training: loss dropped to 0.15 (vs 0.38 in Run 1) but validation degraded severely. The model over-fragmented clusters (NewsWCL50: 409 predicted vs 145 gold clusters), producing near-singleton predictions. The low `ms-lambda=0.3` pushed coreferent mentions apart, and `difficulty-alpha=0.3` drained the training pool too aggressively (65% of chains dropped vs 32% in Run 1).

**Run 3** isolated the single most relevant change: `tau=1.5`. This improved NewsWCL50 by +1.4 pp over Run 1 (0.3149 vs 0.3012 pd_LEA F1) while keeping the overall avg within 1.1 pp (0.4178 vs 0.4290). The optimal clustering threshold shifted from 0.5 to 0.4, indicating the embedding space became tighter under higher PD emphasis.

---

## Cross-dataset performance

A key motivation for this work is training a single model that generalizes across diverse CDCR datasets. The 12 validation datasets span different annotation schemes, entity types, document genres, and chain structures. The following results are from Run 3 (tau=1.5, best checkpoint epoch 42, threshold 0.4, avg pd_LEA F1 = 0.4099).

### Per-dataset results

| Dataset | Val mentions | pd_LEA F1 | LEA F1 | Precision | Recall | Gold clusters | Pred clusters |
|---------|-------------|-----------|--------|-----------|--------|---------------|---------------|
| WECEng | 1,250 | **0.8703** | 0.8914 | 0.9686 | 0.7902 | 233 | 282 |
| HyperCoref | 5,865 | **0.5684** | 0.6045 | 0.4946 | 0.6645 | 3,904 | 3,446 |
| GVC | 977 | 0.4810 | 0.4665 | 0.4324 | 0.5418 | 271 | 190 |
| CEREC | 2,094 | 0.4702 | 0.4605 | 0.4157 | 0.5411 | 422 | 327 |
| FCC | 750 | 0.4656 | 0.4673 | 0.5909 | 0.3842 | 136 | 201 |
| NP4E | 1,744 | 0.4604 | 0.5243 | 0.3605 | 0.6368 | 199 | 145 |
| NewsWCL50 | 2,740 | 0.3230 | 0.3261 | 0.2637 | 0.4169 | 145 | 212 |
| MEANTIME | 1,895 | 0.2563 | 0.3277 | 0.1549 | 0.7432 | 756 | 174 |
| CD2CR | 1,820 | 0.2526 | 0.3124 | 0.2139 | 0.3084 | 1,245 | 789 |
| NIdent | 3,180 | 0.2404 | 0.3334 | 0.1655 | 0.4393 | 800 | 266 |
| ECBplus | 2,710 | 0.2307 | 0.3030 | 0.1386 | 0.6874 | 737 | 165 |
| ECBplusMETAm | 2,619 | 0.2204 | 0.2717 | 0.1395 | 0.5242 | 719 | 207 |

The performance spread is enormous: from 0.87 (WECEng) to 0.22 (ECBplusMETAm), a gap of 65 percentage points with a single model and threshold.

### Performance tiers

**Tier 1 -- Strong (pd_LEA F1 > 0.45): WECEng, HyperCoref, GVC, CEREC, FCC, NP4E**

These datasets share characteristics that align well with the MS Loss metric learning approach:

- Entities tend to have at least some consistent naming across documents (proper nouns, organization names)
- Chain sizes are moderate, allowing the embedding space to capture within-chain variation
- The document contexts provide sufficient signal for disambiguation within the 128-token window

**Tier 2 -- Moderate (0.30-0.45): NewsWCL50**

NewsWCL50 sits alone in the middle: harder than the first tier due to extreme phrasing diversity and metonymic references, but the model still captures enough signal to outperform random clustering significantly.

**Tier 3 -- Weak (pd_LEA F1 < 0.30): MEANTIME, CD2CR, NIdent, ECBplus, ECBplusMETAm**

These datasets present challenges the model systematically struggles with, discussed below.

### Why WECEng is the top performer

WECEng (Wikipedia Event Coreference, English) achieves pd_LEA F1 = 0.87 -- nearly double the average. Several factors explain this:

1. **Wikipedia's encyclopedic style**: Entities are typically introduced by their full name and consistently referred to throughout. "Barack Obama" will appear as "Obama", "the president", but rarely through abstract metonymy like "the White House" (contrast with NewsWCL50's political news style).

2. **Clean entity structure**: Wikipedia articles tend to be well-structured with clear entity boundaries. The 128-token context window captures sufficient disambiguation signal.

3. **Moderate chain diversity with clear anchors**: WECEng chains have phrasing diversity (the PD-weighted training helps), but each chain typically has a dominant canonical form that serves as an embedding anchor. The model clusters around these anchors reliably.

4. **Balanced cluster prediction**: 233 gold vs 282 predicted clusters shows slight over-splitting rather than the severe fragmentation seen in harder datasets. The model produces nearly the right number of clusters.

### HyperCoref's outsized training influence

HyperCoref is the largest dataset in the training corpus by a significant margin:

| Dataset | Training mentions | Training chains | % of total chains |
|---------|-------------------|-----------------|-------------------|
| HyperCoref | ~38,000 (est.) | ~5,400 | ~34% |
| All others | ~90,000 | ~10,300 | ~66% |

Despite the `log(dataset_chain_count)` damping factor in the PK sampler's chain weights (which reduces HyperCoref's per-chain weight by `1/log(5400) ≈ 0.12`), its sheer volume means HyperCoref chains appear in a substantial fraction of training batches. This creates a training dynamic where:

- The embedding space is strongly shaped by HyperCoref's characteristics (hyperlink-derived entity structure, web document style)
- HyperCoref achieves good validation performance (pd_LEA F1 = 0.57) -- respectable but not exceptional given its training dominance
- Datasets with very different characteristics (formal news, event-centric annotations) are relatively underrepresented

The difficulty annealing partially mitigates this: as HyperCoref chains become "easy" (low loss), they drop out of the active training pool (from 8,859 active chains at epoch 1 to ~3,700 by epoch 35), allowing harder chains from other datasets to receive more attention.

### Why the bottom-tier datasets struggle

**ECBplus and ECBplusMETAm** (pd_LEA F1 ≈ 0.22-0.23): These datasets focus on **event coreference** rather than entity coreference. Mentions like "the attack", "the explosion", "the incident" refer to events with very different surface forms. The model's entity-centric training (RoBERTa + mention span embedding) is less suited to capturing event identity, which often depends on temporal and causal reasoning rather than lexical similarity. The extreme over-merging (737 gold clusters collapsed to 165 predicted) confirms the model cannot distinguish between different events described with similar language.

**MEANTIME** (pd_LEA F1 = 0.26): A multilingual dataset focused on events in news. Similar event-centric challenges as ECBplus, compounded by a smaller training set and diverse annotation conventions. The model produces only 174 clusters for 756 gold clusters -- severe over-merging driven by the event ambiguity problem.

**CD2CR** (pd_LEA F1 = 0.25): With 1,245 gold clusters, CD2CR has a very fine-grained annotation scheme where many entities that other datasets would merge are kept separate. The model produces 789 clusters -- closer to the gold count than most datasets, but the precision (0.21) indicates it's creating the wrong clusters rather than too few.

**NIdent** (pd_LEA F1 = 0.24): Focused on named entity identity across documents. Despite the entity-centric focus aligning with the model's design, the severe over-merging (800 gold to 266 predicted) suggests the model cannot distinguish between different entities that share similar names or contexts across documents.

### Does multi-dataset training help generalization?

The multi-dataset approach produces a **single embedding space** that achieves non-trivial performance on all 12 datasets simultaneously. This is a meaningful result: the model learns to encode mentions in a way that captures coreference signal across very different annotation schemes, document genres, and entity types.

However, the generalization is uneven. The model generalizes well to datasets where:
- Entity identity is primarily signaled by **lexical and contextual similarity** (WECEng, HyperCoref, GVC)
- The annotation scheme aligns with **entity-centric coreference** (proper nouns, organizations, people)
- The 128-token context window provides **sufficient disambiguation signal**

The model generalizes poorly to datasets where:
- Coreference requires **event-level reasoning** (ECBplus, ECBplusMETAm, MEANTIME)
- Entities are distinguished by **fine-grained contextual differences** rather than surface form (CD2CR, NIdent)
- Mentions are heavily **metonymic or abstract** (NewsWCL50's country and miscellaneous types)

The PD-weighted sampling with `tau=1.5` shifts the training distribution toward high-diversity chains, which helps datasets like NewsWCL50 and WECEng that benefit from better representation of lexically diverse chains. But it cannot fundamentally address the event-coreference gap (ECBplus family) or the fine-grained entity distinction problem (CD2CR, NIdent), which require different inductive biases rather than different training data distributions.

---

## NewsWCL50 error analysis

The following analysis uses the Run 3 checkpoint (epoch 42, tau=1.5) evaluated at the best threshold (0.4).

### Threshold sensitivity

Both val and test show an extremely sharp threshold cliff:

| Split | Threshold 0.3 | Threshold 0.4 (best) | Threshold 0.5 | Threshold 0.6 |
|-------|---------------|----------------------|---------------|---------------|
| Val (145 gold) | P=0.31, R=0.21, pred=712 | **P=0.27, R=0.39, pred=244** | P=0.17, R=0.65, pred=58 | P=0.07, R=0.96, pred=9 |
| Test (288 gold) | P=0.33, R=0.14, pred=1114 | **P=0.27, R=0.32, pred=404** | P=0.16, R=0.67, pred=98 | P=0.09, R=0.93, pred=15 |

Moving the threshold by just 0.1 in either direction causes a dramatic shift: from severe over-splitting (712 clusters at 0.3) to near-total collapse (9 clusters at 0.6). This indicates the cosine similarity distribution has a narrow transition zone around 0.4-0.5 between coreferent and non-coreferent mention pairs, leaving very little margin for the clustering step.

### Per-topic breakdown

NewsWCL50 covers politically-charged news events with high entity ambiguity. Topics are numbered 0-3 (val) and 4-9 (test).

**Val split (threshold 0.4):**

| Topic | Subtopic (inferred) | Mentions | Chains | pd_LEA F1 | % chains split | % chains merged |
|-------|---------------------|----------|--------|-----------|----------------|-----------------|
| 3 | Democrats sue Russia/Trump | 878 | 45 | **0.2749** | 57.8 | 93.3 |
| 2 | North Korea nuclear announcement | 684 | 28 | 0.3424 | 64.3 | 96.4 |
| 0 | CIA Director meets North Korea | 545 | 33 | 0.3520 | 66.7 | 93.9 |
| 1 | Comey memo | 633 | 39 | 0.3555 | 56.4 | 94.9 |

**Test split (threshold 0.4):**

| Topic | Mentions | Chains | pd_LEA F1 | % chains split | % chains merged |
|-------|----------|--------|-----------|----------------|-----------------|
| 6 | 750 | 54 | **0.2348** | 42.6 | 96.3 |
| 9 | 582 | 39 | 0.2641 | 71.8 | 94.9 |
| 7 | 697 | 61 | 0.3063 | 44.3 | 90.2 |
| 5 | 485 | 43 | 0.3063 | 44.2 | 86.0 |
| 4 | 710 | 52 | 0.3459 | 51.9 | 88.5 |
| 8 | 567 | 39 | 0.3490 | 46.2 | 92.3 |

Merge errors are near-universal (86-97% of chains participate in at least one impure predicted cluster), while split errors affect 42-72% of chains. This pattern holds across all topics, suggesting the problem is systemic rather than topic-specific.

### Per-mention-type breakdown

NewsWCL50 annotates semantic entity types rather than syntactic types (NAM/NOM/PRO):

**Val split (threshold 0.4):**

| Type | Description | Mentions | pd_LEA F1 | Precision | Recall | % in split chains |
|------|-------------|----------|-----------|-----------|--------|-------------------|
| COU | Country | 451 | **0.4279** | 0.6270 | 0.3247 | 98.7 |
| MIS | Miscellaneous/abstract | 254 | **0.4385** | 0.7235 | 0.3146 | 100.0 |
| GRO | Group | 169 | 0.5120 | 0.7028 | 0.4027 | 98.2 |
| EVE | Event | 320 | 0.5616 | 0.7169 | 0.4616 | 100.0 |
| ACT | Actor/person | 1,179 | 0.6121 | 0.5957 | 0.6294 | 88.3 |
| OBJ | Object/artifact | 307 | 0.6415 | 0.7340 | 0.5697 | 100.0 |
| ORG | Organization | 60 | **0.9894** | 1.0000 | 0.9791 | 51.7 |

**Test split (threshold 0.4):**

| Type | Mentions | pd_LEA F1 | Precision | Recall |
|------|----------|-----------|-----------|--------|
| COU | 887 | **0.4181** | 0.6546 | 0.3072 |
| MIS | 411 | **0.4354** | 0.7964 | 0.2996 |
| ACT | 1,168 | 0.5293 | 0.4749 | 0.5979 |
| EVE | 327 | 0.5446 | 0.8250 | 0.4064 |
| GRO | 377 | 0.6196 | 0.8080 | 0.5024 |
| OBJ | 461 | 0.6426 | 0.6722 | 0.6155 |
| ORG | 160 | **0.7207** | 0.7810 | 0.6691 |

Key patterns:

- **COU (Country) and MIS (Miscellaneous) are the hardest entity types** (pd_LEA F1 ~0.42-0.44). Country entities like "USA" appear as "America", "the White House", "Congress", "the Department", "Washington", etc. The model struggles to link these highly metonymic references. Miscellaneous entities (abstract concepts like "Russian interference" or "denuclearization") have even more diverse surface forms.
- **ORG is by far the easiest** (val: 0.99, test: 0.72). Organizations like "WikiLeaks" or "F.B.I." have relatively stable naming conventions across documents, making them easier to cluster.
- **ACT (Actor/person) is the largest category** (1,179 val, 1,168 test) and shows moderate performance (0.53-0.61). The precision-recall balance is more even for actors, suggesting the model handles some name variations well (e.g., "Trump" / "President Trump") but fails on more diverse references ("the defendant", "the nominee").
- **Almost all entity types have 85-100% of their mentions in split chains**, confirming that fragmentation is universal, not confined to specific types.

### Chain-level failure analysis

The error analysis tracks two types of errors per gold chain:

- **Split error**: the chain's mentions end up in more than one predicted cluster (fragmentation)
- **Merge error**: the predicted clusters containing this chain's mentions also contain mentions from other gold chains (contamination)

**Overall error rates (non-singleton chains, val split):**
- 88 / 126 chains (69.8%) have split errors
- 137 / 126 chains (108.7%) have merge errors (>100% because one chain can participate in multiple impure clusters)

**Worst chains by fragmentation (val split, threshold 0.4):**

| Chain | Topic | Size | Split count | Purity | Head words |
|-------|-------|------|-------------|--------|------------|
| 3_USA_Democrats_GROUP | DemocratsSueRUTrump | 110 | 35 | 0.22 | Committee, D.N.C., DNC, Democrats, Party, Supporters |
| 3_RUS_interference_MISC | DemocratsSueRUTrump | 97 | 34 | 0.22 | COLLUSION, abetting, act, actions, activities, allegations |
| 3_Lawsuit_ACTION | DemocratsSueRUTrump | 124 | 31 | 0.28 | DESPERATION, accusations, allegations, assault, attempt |
| 3_USA_Trump_ACTOR | DemocratsSueRUTrump | 70 | 31 | 0.29 | President, Trump, defendant, nominee, president |
| 0_USA_COUNTRY | CIADirectorMeetsNK | 45 | 23 | 0.11 | America, C.I.A., Center, Command, Committee, Department |

**Worst chains by fragmentation (test split, threshold 0.4):**

| Chain | Topic | Size | Split count | Purity | Head words |
|-------|-------|------|-------------|--------|------------|
| 8_USA_Trump_ACTOR | - | 106 | 36 | 0.11 | Trump, client, considered, president, suspect, target |
| 6_USA_COUNTRY | - | 83 | 33 | 0.16 | Cabinet, Committee, Congress, Customs, Department |
| 6_Migrant_caravan_GROUP | - | 185 | 30 | 0.55 | Caravan, Caravans, Families, Immigrants, Members, Migrants |
| 4_USA_COUNTRY | - | 51 | 26 | 0.18 | America, Congress, Department, House, Senate, States |

Interpreting severity through **fragmentation rate** (split_count / chain_size):

| Severity tier | Frag rate | Purity | Example | Interpretation |
|---------------|-----------|--------|---------|----------------|
| Near-total | > 0.4 | < 0.3 | 0_USA_COUNTRY (0.51) | Model treats each mention as unrelated |
| Severe | 0.3-0.4 | 0.2-0.4 | 3_USA_Trump_ACTOR (0.44) | A small core cluster exists but most mentions are lost |
| Moderate | 0.15-0.3 | 0.4-0.6 | 6_Migrant_caravan_GROUP (0.16) | Chain is partially intact with peripheral losses |

---

## Why NewsWCL50 remains a hard challenge

### 1. Extreme phrasing diversity within chains

NewsWCL50 was specifically designed to test coreference resolution under high lexical diversity. Unlike datasets where coreferent mentions share surface forms (e.g., "Barack Obama" / "Obama"), NewsWCL50 chains include:

- **Metonymic references**: "USA" = "the White House" = "Congress" = "the Department of Justice" = "Washington"
- **Role-based references**: "Trump" = "the President" = "the defendant" = "the nominee" = "the Republican candidate"
- **Abstract event descriptions**: "Russian interference" = "COLLUSION" = "the assault on democracy" = "meddling"

The model must learn that "the Department" and "America" can refer to the same entity in context, which requires deep discourse understanding beyond surface-level similarity. The worst chain (`0_USA_COUNTRY`, purity=0.11) demonstrates this: "America", "C.I.A.", "Center", "Command", "Committee", "Department", "House", "Senate" are all valid references to the USA, but the model fragments them into 23 separate clusters.

### 2. Large chain sizes amplify errors

NewsWCL50 has significantly larger coreference chains than most CDCR datasets. Chains with 70-185 mentions span dozens of documents. As chain size grows:

- The number of distinct surface forms increases, making it harder for the model to link all variants.
- The probability of at least one mention being misclassified increases combinatorially.
- A single missed link between two sub-clusters fragments the chain permanently in agglomerative clustering (it cannot recover from early errors).

The `3_Lawsuit_ACTION` chain has 124 mentions with head words including "DESPERATION", "accusations", "allegations", "assault", "attempt" -- an extremely diverse set where no single surface form dominates.

### 3. Cross-document context mismatch

Within a single document, coreference is aided by discourse structure (proximity, syntactic cues, entity tracking). Across documents, these cues vanish. The model must rely purely on semantic similarity between mention embeddings, which are computed from a 128-token window. For mentions like "the investigation" or "the hunt", the local context may not contain enough signal to determine which specific investigation is being referenced.

### 4. Ambiguity between entities within the same topic

NewsWCL50 topics involve multiple overlapping entities: "Trump" (actor) vs "Trump campaign" (event) vs "USA" (country) are three different coreference chains that frequently appear in the same documents with similar contexts. The model must distinguish between these while also linking diverse mentions of each. This creates a precision-recall tension: being more aggressive at merging helps recall for fragmented chains but hurts precision by merging distinct-but-related entities.

This tension is visible in the threshold sensitivity: at 0.4, many chains are still fragmented (split errors: 70%); lowering the threshold to 0.5 reduces fragmentation but causes massive over-merging (predicted clusters drop from 244 to 58 for val). There is no threshold that simultaneously solves both problems.

### 5. Entity type difficulty varies dramatically

The pd_LEA F1 ranges from 0.42 (COU) to 0.99 (ORG) on val, a spread of 57 percentage points. Country and miscellaneous entities fail because:

- Countries are referred to metonymically through their institutions ("the White House", "the Kremlin", "Congress")
- Miscellaneous entities are abstract concepts with no canonical name ("denuclearization", "the nuclear deal", "the agreement")
- These types have high intra-chain diversity and low inter-chain distinctiveness

Organizations succeed because they have stable, distinctive names ("WikiLeaks", "F.B.I.") that the RoBERTa encoder can match reliably.

### 6. The clustering step is a bottleneck

Agglomerative clustering with a fixed threshold is a hard decision boundary. Once two mention groups are not linked at threshold t, they can never be joined. This makes the system brittle to the "weakest link" problem: if a chain has 100 mentions where 98 are within distance t of each other, but 2 peripheral mentions (e.g., "the hunt" for "the investigation") are at distance t+0.05, the chain gets split. A more flexible clustering approach (e.g., iterative merging with learned thresholds, or graph-based community detection) could potentially address this brittleness.

---

## Summary of findings

1. **PD-weighted sampling (tau=1.5) improves NewsWCL50** by +1.4 pp pd_LEA F1 over the baseline, confirming that emphasizing high-diversity chains during training helps the model generalize to lexically diverse coreference.

2. **The improvement is modest** because the core challenge is not training data distribution but the fundamental difficulty of linking metonymic and role-based references across documents. The model would need deeper discourse understanding (beyond 128-token windows) and possibly explicit entity knowledge to bridge references like "the Department" and "America".

3. **Country (COU) and miscellaneous (MIS) entity types are the primary failure modes**, accounting for the lowest per-type pd_LEA F1 scores. These types have the highest intra-chain lexical diversity and the most metonymic reference patterns.

4. **Chain fragmentation (split errors) is the dominant error type**, affecting ~70% of non-singleton chains. The model creates small, pure sub-clusters rather than one large complete cluster -- it under-merges rather than over-merges at the optimal threshold.

5. **The sharpest gains would come from improving cross-document linking of metonymic references** -- teaching the model that "Congress", "the White House", and "Washington" can all refer to the same country entity in context. This may require architectural changes (longer context windows, document-pair encoding) or external knowledge integration rather than further tuning of the training sampling strategy.
