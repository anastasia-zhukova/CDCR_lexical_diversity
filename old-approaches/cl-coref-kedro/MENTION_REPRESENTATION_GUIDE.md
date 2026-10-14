# Mention String Representation Guide

## Overview

This guide explains how mention string representations are created for tokenization and training.

## Purpose

Create string representations that will be tokenized with **128 wordpiece tokens maximum**, combining:
1. **Document context**: 8 sentences from the document (excluding mention's sentence)
2. **Mention context**: The mention's sentence with special markers `[E]` and `[/E]`

## Token Limit: 128 Tokens

The tokenizer will process with:
```python
tokenizer(
    text=document_context,           # 8 sentences
    text_pair=mention_context,       # Mention sentence with [E] [/E]
    max_length=128,
    truncation="only_first"          # Truncates document context first
)
```

**Truncation strategy:**
1. **First**: Remove tokens from end of **document context** (8 sentences)
2. **Last resort**: Remove tokens from end of **mention sentence** (if context + mention > 128)
3. **Critical**: `[E]` and `[/E]` markers must always survive

## String Representation Structure

### Part 1: Document Context (8 sentences)

**Selection strategy:**
- Start with sentence positions: `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- Remove the mention's sentence from this list
- Take remaining 8 sentences
- Group as 4 pairs: `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`

**Example:**
- If mention is in sentence 5 → use sentences `[0,1,2,3,4,6,7,8]`
- If mention is in sentence 9 → use sentences `[0,1,2,3,4,5,6,7]` (remove last)

**Why 8 sentences?**
> "Increased to 4 [pairs] to minimize padding and replace it with tokens from the next sentence"

- More sentences = fill up the 128 token budget more effectively
- Tokenizer with `truncation="only_first"` automatically cuts to fit
- Maximizes information density while respecting token limit

### Part 2: Mention Context (with special tokens)

**Marker placement:**
1. Extract all tokens from the mention's sentence
2. Find mention span: `lower_limit = min(tokens_number)`, `upper_limit = max(tokens_number)`
3. Insert `[/E]` after the mention at position `upper_limit + 1`
4. Insert `[E]` before the mention at position `lower_limit`

**Result:**
```
"token token [E] mention tokens here [/E] token token ..."
```

## Output Format

### mention_representations.parquet

**Columns:**
- `mention_id`: Unique identifier
- `string_representation`: Full string with `[SEP]` separator
- `document_context`: Just the 8 sentences (for inspection)
- `mention_context`: Just the mention sentence with markers (for inspection)

**Format:**
```
string_representation = "{document_context} [SEP] {mention_context}"
```

**Example:**
```
"Sentence 0 tokens. Sentence 1 tokens. Sentence 2 tokens... [SEP] Tokens here [E] the mention [/E] and more tokens."
```

## Why Parquet for This File?

Unlike `mentions_for_pairs.csv`, this uses **Parquet** because:
- Contains `[SEP]` tokens that would interfere with CSV separators
- Has long text strings (up to 128 tokens worth)
- Special characters in mention text
- Needs reliable storage without escaping issues

## Processing Flow

```
mentions_for_pairs.csv
    ├─ mention_id
    ├─ conll_doc_key
    ├─ sent_id
    ├─ tokens_number
    └─ ...
    ↓
Load documents from all_documents.parquet
    ↓
Extract document's sentences → {doc_key: {sent_id: {token_id: token}}}
    ↓
For each mention:
    ├─ Create document context (8 sentences, exclude mention's)
    ├─ Create mention context (insert [E] and [/E])
    └─ Combine: "doc_context [SEP] mention_context"
    ↓
mention_representations.parquet
    ├─ mention_id
    ├─ string_representation
    ├─ document_context
    └─ mention_context
```

## Required Fields in mentions_for_pairs

To create string representations, we need:

| Field | Purpose |
|-------|---------|
| `mention_id` | Unique identifier |
| `conll_doc_key` | To find the document |
| `sent_id` | To find mention's sentence |
| `tokens_number` | To place `[E]` and `[/E]` markers |

That's why we updated `select_required_columns` to include `sent_id` and `tokens_number`.

## Token Budget Breakdown

**Total: 128 tokens**
```
[CLS]  doc_context_tokens...  [SEP]  [E] mention_tokens [/E]  [SEP]  [PAD]...
  1    variable (truncated)     1              variable           1    variable

  └─ Special tokens: 4 (CLS, SEP, SEP, PAD...)
  └─ Available for content: 124 tokens
      ├─ Document context: Truncated to fit
      └─ Mention context: Preserved (with markers)
```

## Example

**Input mention:**
- `mention_id`: "1_10ecb_PER123_2_5_m100"
- `conll_doc_key`: "1/1ecb/10"
- `sent_id`: 2
- `tokens_number`: [5, 6, 7]

**Document sentences:**
```
0: ["This", "is", "sentence", "zero", "."]
1: ["Sentence", "one", "has", "text", "."]
2: ["Tara", "Reid", "checked", "herself", "into", "Promises", "Treatment", "Center", "."]
3: ["Sentence", "three", "continues", "."]
...
```

**Output:**

**document_context:**
```
"This is sentence zero . Sentence one has text . Sentence three continues . ..."
(sentences 0, 1, 3, 4, 5, 6, 7, 8 concatenated)
```

**mention_context:**
```
"Tara Reid [E] checked herself into [/E] Promises Treatment Center ."
(sentence 2 with markers at positions 5-7)
```

**string_representation:**
```
"This is sentence zero . Sentence one has text . Sentence three continues . ... [SEP] Tara Reid [E] checked herself into [/E] Promises Treatment Center ."
```

## Next Steps

After creating string representations:
1. **Tokenization**: Convert strings to token IDs with the 128-token limit
2. **Pair generation**: Create positive/negative pairs for contrastive learning
3. **Training**: Use tokenized representations for model training

## Configuration

Edit `conf/base/parameters_02_pairs_creation.yml`:
```yaml
special_mention_tokens:
  - "[E]"     # Start token
  - "[/E]"    # End token
```

## Running

```bash
kedro run --pipeline 02_pairs_creation
```

This will create `data/03_primary/mention_representations.parquet` with string representations ready for tokenization.
