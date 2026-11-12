# RAG V2 — Team Evaluation Report

## 1. Team

- Team members:  
  - **Sultan AMIOUR**
  - **Muhammad Saad JAVED** 
- Repository: **https://github.com/Msaadjaved/rag-v2-document-qa.git**
- Release tag: `v2` (created locally; push after team review)

## 2. Project and corpora

The application provides retrieval-grounded Q&A over two local corpora:

- **Kestrel KB:** five course-provided Markdown documents and 10 official questions.
- **Own corpus:** the Vellora Duo espresso-machine manual and 12 team questions.

Question sets:
- `data/questions/kestrel.jsonl`
- `data/questions/own.jsonl`

Both corpora were evaluated with configurable chunk sizes, overlaps and top-k retrieval.

## 3. No-context baseline

The no-context baseline was successfully executed for both corpora. It asked the LLM to answer questions without retrieved document context.

Saved results:

| Corpus | Result file | Questions |
|---|---|---:|
| Kestrel | `data/results/kestrel-no-context-20261103-154342-486588100.json` | 10 |
| Own | `data/results/own-no-context-20261103-154229-668139900.json` | 12 |

### Baseline observations

The model sometimes produced plausible general-knowledge answers without access to the documents. These answers were not automatically considered correct.

Examples:
- Kestrel payment and refund questions received generic policy explanations without reliable confirmation of the actual Kestrel policy.
- Kestrel SSO and account-recovery questions received generic suggestions rather than verified product-specific procedures.
- Own-corpus questions about the Vellora Duo's error codes, exact settings, warranty and specifications often received unknown or speculative answers.
- The model gave a generic milk-temperature recommendation in the own-corpus baseline rather than the manual's specific recommendation of approximately 5°C.

The baseline outputs were reviewed against the corpus-specific evidence. Generic or plausible answers were not treated as verified answers.

**Question-set decision:** No question was removed solely because the no-context model produced a plausible response. The questions test document-specific facts, procedures and policies. The team should confirm this decision before submission and replace any question that it determines is correctly answerable without retrieval.

## 4. Retrieval evaluation

### 4.1 Method

The evaluation used gold evidence from the question records. A retrieved chunk is considered relevant when it matches the expected evidence quote. Unanswerable questions are excluded from the retrieval HitRate denominator.

The evaluation reports binary HitRate@1, @3, @5 and @10, both overall and by question type.

A hit means at least one relevant chunk appears within the specified top-k results. This is not the same as counting the proportion of all relevant chunks retrieved.

The final selected retriever was Sentence Transformers using `all-MiniLM-L6-v2`. TF-IDF was also used in configuration comparison experiments.

### 4.2 Configuration comparison

| Corpus | Configuration | Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---|---|---:|---:|---:|---:|
| Kestrel | 800 / 10 / 10 | TF-IDF | 62.5% | 87.5% | 87.5% | 100% |
| Kestrel | 500 / 20 / 10 | TF-IDF | 25% | 75% | 75% | 100% |
| Own | 800 / 10 / 10 | TF-IDF | 66.67% | 83.33% | 91.67% | 91.67% |
| Own | 500 / 20 / 10 | TF-IDF | 66.67% | 91.67% | 100% | 100% |

Chunk-size and overlap values are absolute configuration values, not percentages. The final value in each configuration is top-k.

### 4.3 Final Kestrel results

Selected configuration: **800 / 10 / 10**

Retriever: Sentence Transformers (`all-MiniLM-L6-v2`)

- Total questions: 10
- Answerable questions: 8

| Metric | Hit rate |
|---|---:|
| HitRate@1 | 62.5% |
| HitRate@3 | 87.5% |
| HitRate@5 | 87.5% |
| HitRate@10 | 100% |

Results by question type:

| Question type | Count | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|---:|
| Exact identifier | 2 | 50% | 50% | 50% | 100% |
| Split answer | 2 | 100% | 100% | 100% | 100% |
| Two-hop | 1 | 100% | 100% | 100% | 100% |
| Versions | 2 | 0% | 100% | 100% | 100% |
| Vocabulary | 1 | 100% | 100% | 100% | 100% |

Saved result:
`data/results/kestrel-sentence-transformers-800-10-10.json`

Generated-answer result:
`data/results/kestrel-sentence-transformers-800-10-10-generated.json`

### 4.4 Final Own Corpus results

Selected configuration: **500 / 20 / 10**

Retriever: Sentence Transformers (`all-MiniLM-L6-v2`)

- Total questions: 12
- Answerable questions: 12

| Metric | Hit rate |
|---|---:|
| HitRate@1 | 66.67% |
| HitRate@3 | 91.67% |
| HitRate@5 | 100% |
| HitRate@10 | 100% |

Results by question type:

| Question type | Count | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|---:|
| Different vocabulary | 3 | 66.67% | 100% | 100% | 100% |
| Exact identifier | 5 | 60% | 80% | 100% | 100% |
| Not-in-corpus trap | 1 | 100% | 100% | 100% | 100% |
| Split answer | 2 | 50% | 100% | 100% | 100% |

Saved result:
`data/results/own-sentence-transformers-500-20-10.json`

Generated-answer result:
`data/results/own-sentence-transformers-500-20-10-generated.json`

### 4.5 Retriever comparison

At the selected configurations, TF-IDF and Sentence Transformers produced the same measured overall hit rates:

- Kestrel, 800/10/10: 62.5%, 87.5%, 87.5%, 100%.
- Own, 500/20/10: 66.67%, 91.67%, 100%, 100%.

This does not establish that both retrievers return identical rankings or chunks. It only establishes that their measured binary hit rates were equal at these cutoffs.

## 5. Manual answer grading

The generated answers were reviewed against the source documents. Grades reflect factual correctness and completeness, not merely whether a relevant chunk was retrieved.

Grading categories:
- **Correct:** supported by the source and answers the question.
- **Partial:** substantially correct but incomplete, ambiguous or containing an unsupported detail.
- **Wrong:** materially contradicts the source or gives an incorrect answer.
- **Unsupported:** asserts information that the retrieved evidence does not establish.

### 5.1 Kestrel

| Question | Grade | Rationale |
|---|---|---|
| KST-01 | Correct | Correctly identifies the KST-2044 connection problem and the IP allowlist correction. |
| KST-02 | Correct | Correctly explains how an administrator can free a seat or an Owner can add seats, and mentions invitation expiry. |
| KST-03 | Correct | Correctly describes the payment retry schedule, grace period and later account restrictions. |
| KST-04 | Correct | Correctly identifies SAML 2.0, eligible plans and the setup procedure. |
| KST-05 | Correct | Correctly describes backup-code recovery, administrator reset and support escalation for a sole administrator. |
| KST-06 | Correct | Correctly applies the current 60-day pro-rated refund policy to a request made after 45 days. |
| KST-07 | Partial | Mentions both the old 15% and current 20% annual discount but does not clearly prioritize the current policy. |
| KST-08 | Wrong | Gives the wrong escalation destination. The expected team and channel are Messaging and `#oncall-messaging`. |
| KST-09 | Correct | Correctly indicates that the corpus does not establish iPhone app support. |
| KST-10 | Correct | Correctly avoids inventing a Business plan seat price and points to the current pricing information. |

**Kestrel grading summary:** 8 correct, 1 partial, 1 wrong.

KST-07 demonstrates a version-selection problem: older and newer policy evidence can both be retrieved, but the generated answer must prioritize the current policy.

KST-08 demonstrates a generation failure despite relevant evidence being available in the retrieved results.

### 5.2 Own Corpus

| Question | Grade | Rationale |
|---|---|---|
| OWN-01 | Correct | Correctly gives the Vellora Duo tank capacity as 1.8 litres. |
| OWN-02 | Correct | Correctly describes the filter preparation and replacement interval. |
| OWN-03 | Correct | Correctly identifies a grind that is too fine or a dirty brewing unit and provides the relevant corrective steps. |
| OWN-04 | Correct | Correctly states that vinegar should not be used and that the recommended descaling liquid should be used. |
| OWN-05 | Correct | Correctly explains the descaling and subsequent rinsing procedure. |
| OWN-06 | Partial | Identifies the foreign object or stone and the unplugging, vacuum and hex-key procedure, but omits the instruction to turn the upper burr anticlockwise. |
| OWN-07 | Correct | Correctly states that the brewing unit should be rinsed weekly. |
| OWN-08 | Partial | Claims that strength level 5 uses exactly 14 grams. The source only specifies a range of 7–14 grams across strength levels 1–5; it does not explicitly assign 14 grams to level 5. |
| OWN-09 | Correct | Correctly gives approximately 5°C and recommends milk straight from the refrigerator. |
| OWN-10 | Correct | Correctly states that at least 10 cm should be left behind the machine. |
| OWN-11 | Correct | Correctly identifies Wi-Fi support using the Wi-Fi and app-related error codes. |
| OWN-12 | Correct | Correctly states that the warranty lasts two years from purchase, with the receipt required. |

**Own Corpus grading summary:** 10 correct, 2 partial, 0 wrong.

OWN-06 is a completeness issue: the relevant evidence was retrieved, but an important procedural detail was omitted.

OWN-08 is an unsupported inference: the retrieved source gives a range but does not establish the exact amount for strength level 5. The answerability annotation should be reviewed if the question expects an exact value rather than the documented range.

### 5.3 Retrieval versus generation failures

The distinction matters:

- If the relevant chunk is absent from the retrieved results, the problem is retrieval.
- If the relevant evidence is retrieved but the answer is incorrect, incomplete or unsupported, the problem is generation or answer grounding.

Examples:
- OWN-06: relevant chunk retrieved at rank 3; incomplete generated procedure.
- OWN-08: relevant chunk retrieved at rank 1; unsupported exact-value inference.
- KST-08: one relevant chunk was retrieved at rank 1, but the second supporting chunk was ranked 6; the generated answer also gave the wrong escalation destination.

## 6. Words Analytics diagnostics

### 6.1 Kestrel: KST-08

Question:

> Error KST-5031 happened on the last two scheduled reports. Which team should support contact, and where?

Words Analytics Question mode observations:

Present tokens and corpus chunk counts:

- `error`: 9 chunks
- `kst-5031`: 2 chunks
- `happened`: 2 chunks
- `on`: 31 chunks
- `the`: 43 chunks
- `last`: 5 chunks
- `two`: 4 chunks
- `scheduled`: 8 chunks
- `reports`: 7 chunks
- `which`: 3 chunks
- `team`: 11 chunks
- `support`: 20 chunks
- `contact`: 6 chunks
- `and`: 43 chunks

Absent tokens:
- `should`
- `where`

Diagnosis:

The main question vocabulary is present. The required information is distributed across more than one document passage. The support runbook passage for KST-5031 appears at rank 1, while the service-catalog passage identifying Messaging and `#oncall-messaging` appears at rank 6.

This is a multi-evidence ranking problem: the first passage is retrieved immediately, but the second supporting passage falls outside the top five. The question achieves HitRate@5 because at least one relevant passage is already present, even though not all evidence needed for a complete answer is available within those five results.

### 6.2 Own Corpus: OWN-01

Question:

> What is the water tank capacity of the Vellora Duo?

Words Analytics Question mode observations with chunk size 800 and overlap 10:

Present tokens and corpus chunk counts:

- `what`: 28 chunks
- `is`: 38 chunks
- `the`: 44 chunks
- `water`: 30 chunks
- `tank`: 15 chunks
- `of`: 42 chunks
- `vellora`: 8 chunks
- `duo`: 3 chunks

Absent token:
- `capacity`

Diagnosis:

The exact word `capacity` is absent from the corpus vocabulary, creating a vocabulary mismatch between the question and the manual. Under the 800/10 configuration, the relevant tank-capacity passage was not retrieved in the top ten.

The smaller 500/20 configuration improved this case: the relevant chunk was retrieved at rank 4. This suggests that chunking configuration can help expose relevant evidence even when the question and source use different wording.

## 7. Configuration decisions

The final configuration decisions are:

| Corpus | Chunk size | Overlap | Top-k | Reason |
|---|---:|---:|---:|---|
| Kestrel | 800 | 10 | 10 | Higher HitRate@1, @3 and @5 than the tested 500/20 configuration. |
| Own | 500 | 20 | 10 | Higher HitRate@3, @5 and @10 than the tested 800/10 configuration. |

Both selected configurations achieved 100% HitRate@10.

These decisions are based on the measured retrieval results. The team should confirm them before submission.

## 8. Known failures and limitations

1. **Citation extraction:** Generated answers contain citation identifiers inside their JSON-formatted answer strings, but the outer `generated_answer.citations` field is empty in the inspected generated result files. Citation parsing or structured-output handling needs correction.

2. **KST-07 policy versioning:** The generated answer did not clearly prioritize the current 2026 annual discount over the older 2025 policy.

3. **KST-08 multi-evidence answer:** The service-catalog evidence appears at rank 6, outside the top five. The generated answer also identifies the wrong escalation destination.

4. **OWN-06 procedural completeness:** The answer omits the instruction to turn the upper burr anticlockwise.

5. **OWN-08 unsupported precision:** The answer claims exactly 14 grams at strength level 5, while the manual only documents a 7–14 gram range across levels 1–5.

6. **Answerability annotation:** OWN-08 should be reviewed because its current wording requests a precise level-specific quantity that the source does not explicitly establish.

7. **No-context hallucination risk:** The baseline sometimes generates plausible generic advice without source evidence. These outputs must not be treated as verified product or policy facts.

8. **Document ingestion:** PDF and DOCX ingestion are not implemented. Supported uploads are UTF-8 TXT and Markdown files.

9. **Evaluation limitations:** Binary HitRate measures whether at least one relevant chunk appears in the top-k results. It does not measure complete evidence coverage for multi-hop questions.

10. **Manual grading:** Subjective answer grades require human review. They are not automatically assigned by the application.

## 9. Result files

Important evaluation artifacts:

- `data/results/kestrel-no-context-20261103-154342-486588100.json`
- `data/results/own-no-context-20261103-154229-668139900.json`
- `data/results/kestrel-sentence-transformers-800-10-10.json`
- `data/results/kestrel-sentence-transformers-800-10-10-generated.json`
- `data/results/own-sentence-transformers-500-20-10.json`
- `data/results/own-sentence-transformers-500-20-10-generated.json`

Configuration-comparison TF-IDF result files are also retained in `data/results/`.

## 10. How to run

See `README.md` for installation, environment configuration, API endpoints and application usage.

Start the application using:

```bash
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

## 11. Release

The `v2` Git tag has been created locally. The repository and tag should be pushed to the team's remote after review and approval.


