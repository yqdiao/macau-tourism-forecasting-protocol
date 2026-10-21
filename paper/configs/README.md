# Recorded configuration evidence

## Qwen3-8B

`qwen8b_manifest.json` transcribes the supplied configuration note into English structured metadata. It includes the model revision and hash, protocol and sampling hashes, backend revision, thinking setting, hardware type, and recorded resource observations. The recorded protocol and sampling hashes were independently checked against compact, sorted-key JSON serialization of their disclosed fields.

This is a transcription of a configuration document, not a replacement for the original machine-generated runtime registry. Host names, GPU UUIDs and the local interpreter path were omitted. No prompt contents, secrets or model weights are included.

The source note's role table conflicts with the audited S3-I/S3-R stage behavior; the inconsistent table is not copied. Strategy interpretation follows the delivery/stage audit described in `paper/data_dictionary.md`. The source note also reports a 16-billion-byte chain stop threshold alongside a 10-billion-byte design budget. The R0 process memory observation does not establish that every simultaneous two-card run stayed within that budget.

The 8B setup uses one local service slot and thinking disabled. M5 in the manuscript refers to the separate 4B-thinking direct-forecast results. Changes in backbone, thinking and output permission are not treated as one isolated factor.

## Lightweight revisions

`r3/` preserves the actual supplied R3 protocol and selection JSON records. Read its README for the missing-source and exploratory-selection boundaries. These records are evidence snapshots, not unexecuted template files.
