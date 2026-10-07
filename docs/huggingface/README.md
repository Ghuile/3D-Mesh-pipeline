# Hugging Face dataset documentation

Source copies of the dataset cards published on Hugging Face. The four repositories are grouped in the [3D Mesh Pipeline | CPSI 2026 collection](https://huggingface.co/collections/Ghuile/3d-mesh-pipeline-cpsi-2026).

| Stage | Card source | Hosted dataset |
|---|---|---|
| Raw baseline FBX | [README](metahuman-data-generation/README.md) | [Generation](https://huggingface.co/datasets/Ghuile/metahuman-data-generation) |
| Sanitized baseline OBJ | [README](metahuman-data-sanitized/README.md) | [Sanitized](https://huggingface.co/datasets/Ghuile/metahuman-data-sanitized) |
| Parsed arrays and baseline outputs | [README](metahuman-data-parsed/README.md) | [Parsed](https://huggingface.co/datasets/Ghuile/metahuman-data-parsed) |
| OOD data and features | [README](unreal-metahuman-stress-test/README.md) | [Stress test](https://huggingface.co/datasets/Ghuile/unreal-metahuman-stress-test) |

Cards were prepared on 7 October 2026 from complete Hub file inventories and representative NPZ array headers. Download examples pin the inspected data revisions; they do not claim a new experimental run or full geometry audit. The documentation publication does not change existing data files or assign a dataset license.

When updating a card, verify counts, schemas, revision IDs, and download patterns against its Hub repository. Keep this source copy and the hosted `README.md` synchronized. The paper citation and scientific limitations should remain consistent with the main repository documentation.
