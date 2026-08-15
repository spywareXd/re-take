# re-take (retake-detection)

## Project Summary
This project aims to build an automated pipeline for detecting and removing retakes, restarts, stutters, and redundancies in spoken transcripts (from video or audio sources). 

When creators record content, they often stumble, correct themselves, or restart a sentence. This tool analyzes transcription data (including metadata like timestamps, speaker IDs, and transcription confidence) to identify these overlapping or corrective segments. Ultimately, this enables an automated editing process that cuts out the "bad" takes and keeps only the final, clean versions.

## Core Objectives
- **Parse raw transcripts**: Ingest JSON transcript data with word/segment level metadata, timestamps, and confidence scores.
- **Detect redundancies**: Use a combination of token matching and semantic similarity to find repeated or corrected phrases within a local window.
- **Identify retake boundaries**: Pinpoint exactly where a mistake was made and where the clean take begins.
- **Automated Decision Making**: Leverage LLMs and advanced heuristics to adjudicate ambiguous overlaps, score confidence, and finalize the "cut" decisions.
- **Video/Audio Editing**: Generate automated edit lists to programmatically trim the original media files (e.g., via FFmpeg).

## Pipeline Architecture
The system is designed around a modular, 8-stage pipeline:
1. **Transcription (`01_transcription`)**: Extracting audio and generating base transcripts with timestamps.
2. **Preprocessing (`02_preprocessing`)**: Cleaning, tokenizing, and chunking the raw text.
3. **Token Matching (`03_token_matching`)**: Finding exact or near-exact word repetitions using n-grams and edit distance.
4. **Semantic Matching (`04_semantic_matching`)**: Discovering conceptually similar retakes using embeddings and vector search (FAISS).
5. **Span Aggregation (`05_span_aggregation`)**: Merging and clustering candidate spans into consolidated retake windows.
6. **LLM Adjudication (`06_llm_adjudication`)**: Utilizing a language model to make the final cut/keep decisions with associated confidence scores.
7. **Review (`07_review`)**: Exporting reports and structured JSON for human review or debugging.
8. **Video Edit (`08_video_edit`)**: Translating the finalized cuts into FFmpeg commands to produce the final, polished media.
