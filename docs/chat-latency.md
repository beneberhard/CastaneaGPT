# Chat latency measurements

The API writes one `CHAT_TIMING` JSON summary per completed or failed chat.
Read the summaries with:

```bash
docker compose logs api | rg CHAT_TIMING
```

The summary includes a random request ID, status, cache counters, context size,
and durations in seconds. The timing summary does not contain prompts,
retrieved excerpts, answers, or credentials.

## Interpreting the measurements

- `request_preparation`: session, stand context, optional NDVI calculations, and retrieval-question assembly.
- `retriever_initialization`: getting the cached retriever, or loading/building it when absent.
- `index_load`: loading the stored index; included in initialization, and absent on a cache hit.
- `retrieval_total`: hybrid document search after initialization.
- `embedding_request`: embedding-library operation, including remote request/retries where applicable.
- `vector_retrieval`: vector search including its embedding operation.
- `keyword_retrieval`: BM25 keyword search.
- `prompt_preparation`: selecting context, glossary matching, and constructing the final prompt.
- `model_to_first_text`: model invocation until the first nonempty text delta.
- `after_first_text`: remaining stream duration, including forwarding deltas.
- `model_stream_total`: the complete model streaming operation.
- `finalization`: citation processing and saving completed conversation memory.

Some durations overlap: do not add embedding, vector, keyword and retrieval
measurements together. Hybrid retrieval may run its branches concurrently.
Likewise, `index_load` is part of `retriever_initialization`, and
`model_stream_total` includes both model-to-first-text and subsequent streaming.

`retriever_cache_before`/`after` identify whether the cached retriever was reused.
A reused retriever does not need to call `get_index()` again, so the index cache's
hit count may stay unchanged while the retriever's hit count increases.

Compare the same short question twice in one API process, without restarting
between requests. Disable session memory for that comparison so the second
question does not receive a longer prompt. A question without NDVI keywords
isolates the core retrieval and generation path; NDVI latency needs a separate
measurement.

## Measured on 2026-11-01

Two consecutive streaming requests in the same API process used the same question,
“What is chestnut ink disease?”, stand 1, English, short verbosity, and no session
memory. Both prompts contained 8 chunks and 17,024 characters.

| Stage (seconds) | First request | Second request |
| --- | ---: | ---: |
| Request preparation | 0.0982 | 0.0022 |
| Retriever initialization, including index loading | 33.3397 | <0.0001 |
| Hybrid retrieval | 35.6697 | 34.7581 |
| Prompt preparation | 0.0312 | 0.0274 |
| Model to first text | 2.6842 | 0.8040 |
| Remaining stream | 1.0475 | 1.0294 |
| Finalization | 0.0002 | 0.0001 |
| Total API time | 72.8712 | 36.6219 |

Index loading accounted for 31.9616 seconds of first-request initialization.
Keyword search took 0.0011 and 0.0019 seconds, respectively; vector retrieval
(including embedding) took 35.6685 and 34.7568 seconds. The second request
increased the retriever cache hit counter from 0 to 1 without another index load.

The initial live measurements did not record embedding duration independently:
LlamaIndex replaced the embedding callback manager while loading the index.
The final instrumentation passes that manager explicitly to index loading so
future requests can report this measurement. The live results above were not
rerun after that correction.

A separate local-only benchmark reused a stored document vector as the query,
with no embedding or model API request. MMR selection of 40 matches from 10,488
vectors (1,536 dimensions each) took **30.110 seconds**, excluding the separate
process's index loading (31.498 seconds). This is a different query vector from
the live test, but it establishes that local MMR alone can account for most of
the recurring vector-retrieval delay.

The next optimization target is local vector retrieval, especially the MMR
implementation that repeatedly compares vectors across the corpus. Consider
shortlisting candidates before MMR or using a vectorized/indexed implementation,
then compare source relevance and response quality. Reducing the final context
from 8 chunks does not address this measured search cost by itself.
