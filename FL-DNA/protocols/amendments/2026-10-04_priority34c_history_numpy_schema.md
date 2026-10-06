# Priority34C — historical NumPy ID schema

Before parser correction/replay. Preserved second metadata-only failure:
Phase2 targets.pt stores source_rows as NumPy ndarray. Accept NumPy arrays by
tolist before the same integer-only validation. This changes no exclusions
semantics, targets, seeds or scientific outcomes. Add path context on parsing
errors. The ambiguous sources key is interpreted as IDs only when it is an
integer sequence; mapping/list-of-path source-code provenance is not a record
ID field. Explicit source_rows/source_ids remain strict and must parse.
No new P34C target records have been accessed and no scientific job launched.
