# LITEWAY

>An extra-lightweight, convolution-only network with a single linear layer for human activity recognition

TL;DR:
LITEWAY demonstrates that replacing recurrent HAR architectures with structured convolutions enables highly efficient temporal modeling, achieving strong accuracy while drastically reducing computation, model size, inference time, and energy consumption for real-world TinyML deployment.

The proposed architectures are implemented as: [**LITEWAY Full** ](models/LITEWAYfull.py) and [**LITEWAY Light** ](models/LITEWAYlight.py)

## LITEWAY Framework

<img src="assets/architectures_both.png" alt="LITEWAY Architecture" width="993"/>



## Results Summary Across 16 HAR Datasets <!-- (see below for details of the individual datasets) -->


| Performance | Hardware (STM32L4S5) |
|:---:|:---:|
| <img src="assets/liteway_performance.png" width="100%"> | <img src="assets/liteway_hardware.png" width="100%"> |


## Details Results

<img  src="assets/results_main_box.png" alt="Results summary" width="993"/>

---
## Experiments and Reproduction

### Datasets

To download the evaluated HAR datasets and view the repository's data structure organization, please refer to the [`datasets/readme`](datasets/readme.md) file.

---

### Training & Evaluation

Run training using:

```bash
python3 train.py --seeds [SEED] --model [MODEL] --dataset [DATASET]

MODEL: liteway, liteway_light
DATASET: hapt, wear, realdisp, uschad, rw, recgym, motionsense, dsads, oppo, oppoloc, dg, uci, pamap2, skodar, mhealth, sho
SEED: 1, 2, 3, 4, 5.
```

<!-- Models:`liteway`, `liteway_light` <br>
Datasets: `hapt`, `wear`, `realdisp`, `uschad`, `rw`, `recgym`, `motionsense`, `dsads`, `oppo`, `oppoloc`, `dg`, `uci`, `pamap2`, `skodar`, `mhealth`, `sho`
-->

#### Example

```bash
python3 train.py --seeds 5 --model liteway --dataset pamap2
```

---

### Acknowledgements

We thank the authors of [TinyHAR](https://doi.org/10.1145/3544794.3558467) open-source repositories for providing useful code that were adapted in this work.
