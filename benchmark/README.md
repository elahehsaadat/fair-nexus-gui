# Metadata-to-NeXus benchmark

This folder is a small, reproducible dataset for testing metadata-to-NeXus path mapping across three techniques. It is intentionally separate from the Streamlit application.

| Technique | Raw cases | Formats/instruments | Reference NeXus files |
| --- | ---: | --- | ---: |
| Electron microscopy (`NXem`) | 3 | TESCAN, FEI Helios, ZEISS SEM TIFF | 3 metadata references |
| XPS (`NXxps`) | 2 | SPECS XML, PHI SPE | 2 |
| Raman (`NXraman`) | 2 | ROD, WITec text | 2 |

`benchmark_db.json` is the machine-readable experiment database. Each record identifies the raw input, target application definition, source snapshot, checksum, and reference.

The XPS and Raman companions are upstream reference NeXus outputs. Each microscopy image has a human-readable `reference.json` containing verified embedded metadata and the current extractor baseline. These microscopy references are not authoritative NeXus outputs and must not be scored as path-level ground truth.

## Intended workflow

```text
raw file -> format-specific metadata extractor -> metadata dictionary
         -> repository mapping logic -> predicted NeXus paths
         -> compare with reference paths
```

Only the three TIFF files should be tested directly in the current GUI. The XPS and Raman inputs require format-specific parsers before their metadata dictionaries are passed to the mapper. Keeping parsing separate ensures that the experiment measures mapping quality rather than parser coverage.

## Provenance

- XPS raw/reference pairs: [FAIRmat-NFDI/pynxtools-xps](https://github.com/FAIRmat-NFDI/pynxtools-xps), pinned to commit `f7eefa1a454507482ac3408cd11793c83fe4edd9` (Apache-2.0).
- Raman raw/reference pairs: [FAIRmat-NFDI/pynxtools-raman](https://github.com/FAIRmat-NFDI/pynxtools-raman), pinned to commit `79a7a4ec1459b8520c0c31b954016937884b81f9` (Apache-2.0).
- FEI and ZEISS TIFF fixtures: [hyperspy/rosettasciio](https://github.com/hyperspy/rosettasciio), pinned to commit `bc254db14cd7d4d23169b11aeb622a0a7eac1fbe` (GPL-3.0 repository license).
- TESCAN TIFF: copied from this repository's existing `sample_data` folder.

The licenses above are the upstream repository licenses. Confirm dataset-specific redistribution and attribution requirements before publishing the benchmark independently.

## Validate

Run from the repository root:

```bash
.venv/bin/python -m json.tool benchmark/benchmark_db.json >/dev/null
sha256sum -c benchmark/SHA256SUMS
.venv/bin/python -m unittest discover -s tests -v
```
