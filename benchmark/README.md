# Metadata-to-NeXus benchmark

This folder is a small, reproducible TIFF-only dataset for testing metadata-to-NeXus path mapping across three microscopy techniques. It is intentionally separate from the Streamlit application.

| Technique | TIFF cases | Formats/instruments | References |
| --- | ---: | --- | ---: |
| Electron microscopy (`NXem`) | 3 | TESCAN, FEI Helios, ZEISS SEM TIFF | 3 metadata references |
| Atomic force microscopy (`NXafm`) | 2 | Rendered AFM TIFF | 2 metadata references |
| Raman microscopy (`NXraman`) | 2 | Derived Raman-map TIFF | 2 metadata references |

`benchmark_db.json` is the machine-readable experiment database. Each record identifies the raw input, target application definition, source snapshot, checksum, and reference.

Each microscopy image has a human-readable JSON reference containing provenance, verified file properties, the current extractor baseline, and limitations. These references are not authoritative NeXus outputs and must not be scored as path-level ground truth.

The TIFF-only subset is organized under `tiff_by_technique/` with its own `tiff_database.json`.

## Intended workflow

```text
TIFF -> metadata extractor -> metadata dictionary
     -> repository mapping logic -> predicted NeXus paths
     -> compare with curated reference concepts
```

All seven TIFF files can be tested directly in the current GUI.

## Provenance

- FEI and ZEISS TIFF fixtures: [hyperspy/rosettasciio](https://github.com/hyperspy/rosettasciio), pinned to commit `bc254db14cd7d4d23169b11aeb622a0a7eac1fbe` (GPL-3.0 repository license).
- TESCAN TIFF: copied from this repository's existing `sample_data` folder.
- Raman-map TIFFs: [ORead15/2DM-Anticounterfeiting](https://github.com/ORead15/2DM-Anticounterfeiting), pinned to commit `ccee018ca2b48160707ab5c0e852134e45f6744c` (Apache-2.0).
- AFM TIFFs and NanoScope profiles: [Zenodo record 3974975](https://zenodo.org/records/3974975), DOI `10.5281/zenodo.3974975` (CC BY 4.0).

The licenses above are the upstream repository licenses. Confirm dataset-specific redistribution and attribution requirements before publishing the benchmark independently.

## Validate

Run from the repository root:

```bash
.venv/bin/python -m json.tool benchmark/benchmark_db.json >/dev/null
sha256sum -c benchmark/SHA256SUMS
.venv/bin/python -m unittest discover -s tests -v
```
