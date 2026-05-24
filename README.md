# LITEWAY

>An extra-lightweight, convolution-only network with a single linear layer for human activity recognition

TL;DR:
LITEWAY demonstrates that replacing recurrent HAR architectures with structured convolutions enables highly efficient temporal modeling, achieving strong accuracy while drastically reducing computation, model size, inference time, and energy consumption for real-world TinyML deployment.


## LITEWAY Framework
<!-- [architectures_both.pdf](https://github.com/user-attachments/files/28199870/architectures_both.pdf) -->
<img width="993" alt="architectures_both" src="https://github.com/user-attachments/assets/92a67067-02c7-4c73-938d-64f11e3be9e3" />


## Results Summary Across 16 HAR Datasets (s. details below)
<!-- [results_summary-1.pdf](https://github.com/user-attachments/files/28199872/results_summary-1.pdf) -->
<img width="445" alt="results_summary-1" src="https://github.com/user-attachments/assets/7fe183da-0b51-40c4-bee1-2373acf1c9d4" />


## Comparison of SOTA and LITEWAY on STM32L4S5
<!-- <img width="4225" height="1778" alt="LITEWAY_energy" src="https://github.com/user-attachments/assets/f753801c-c05c-4447-ab93-4d50cc5215aa" /> -->

| Model         | Inf. Time (ms) | Weight (KiB) | Activation (KiB) | Cycles/MAC | CPU (% load) | Energy (mJ/Inf) |
|---------------|----------------|--------------|------------------|-------------|---------------|-----------------|
| TinyHAR       | 249.01 | 107.48 | 62.70 | 13.51 | 24 | 19.14 |
| MLPHAR        | 114.81 | 342.41 | 40.54 | 11.99 | 11 | 8.73 |
| TinierHAR     | 81.42 | 39.54 | 14.43 | 25.58 | 8 | 6.36 |
| **LITEWAY-F** | 56.71 | 10.63 | 16.35 | 33.86 | 5 | 4.35 |
| **LITEWAY-L** | 37.44 | 10.07 | 16.03 | 30.54 | 3 | 2.90 |


## Details Results
<!-- [results_main_box.pdf](https://github.com/user-attachments/files/28199876/results_main_box.pdf) -->
<img width="993" alt="Details results with 16 HAR datasets" src="https://github.com/user-attachments/assets/7859758e-7310-443f-a4d6-e8b7d4521682" />

---

## Datasets

The evaluated HAR datasets can be downloaded using the links provided in the [`datasets/`](datasets) directory.

---

## Training & Evaluation

Run training using:

```bash
python3 train.py --seeds [SEED] --model [MODEL] --dataset [DATASET]
```

### Available Models

- `liteway`
- `liteway_light`

### Example

```bash
python3 train.py --seeds 5 --model liteway --dataset pamap2
```

---

## License

LITEWAY is released under the MIT License. See the [LICENSE](LICENSE) file for details.