# TIFF benchmark by technique

This is the TIFF-only subset of the repository benchmark. The folders represent the scientific technique established by the source provenance; the `.tif` extension alone does not identify a technique.

| Folder | TIFF files | Technique | Reference type |
| --- | ---: | --- | --- |
| `sem/` | 3 | SEM/FIB-SEM (`NXem`) | Curated embedded-metadata baselines |
| `raman_microscopy/` | 2 | Raman microscopy (`NXraman`) | Source-backed derived-image baselines |
| `afm/` | 2 | Atomic force microscopy (`NXafm`) | Source-backed rendered-image baselines and NanoScope profiles |

Every TIFF has a neighboring `reference.json` or `.reference.json`. The AFM images also include the upstream NanoScope profile text. References document provenance, checksums through the parent manifest, verified image properties, current extractor behavior, and limitations.

The SEM TIFFs contain vendor metadata that the current application decodes. The Raman and AFM TIFFs are rendered RGB exports with generic TIFF fields; their scientific context is retained in the reference files rather than embedded in the images.

See `tiff_database.json` for the machine-readable TIFF-only case list and `../benchmark_db.json` for the complete mixed-format benchmark.
