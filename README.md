
# liteway

An extra-lightweight, convolution-only network with a single linear layer for human activity recognition


## LITEWAY framework
LITEWAY demonstrates that replacing recurrent HAR architectures with structured convolutions enables highly efficient temporal modeling, achieving strong accuracy while drastically reducing computation, model size, inference time, and energy consumption for real-world TinyML deployment.
[architectures_both.pdf](https://github.com/user-attachments/files/28199870/architectures_both.pdf)
<img width="2980" height="1692" alt="architectures_both" src="https://github.com/user-attachments/assets/92a67067-02c7-4c73-938d-64f11e3be9e3" />


[results_summary-1.pdf](https://github.com/user-attachments/files/28199872/results_summary-1.pdf)
<img width="445" height="161" alt="results_summary-1" src="https://github.com/user-attachments/assets/7fe183da-0b51-40c4-bee1-2373acf1c9d4" />

<img width="4225" height="1778" alt="LITEWAY_energy" src="https://github.com/user-attachments/assets/f753801c-c05c-4447-ab93-4d50cc5215aa" />

## Comparison of SOTA and LITEWAY on STM32L4S5

| Model | Inf. Time (ms) | Weight (KiB) | Activation (KiB) | Cycles/MAC | CPU (% load) | Energy (mJ/Inf) |
|-------|----------------|--------------|------------------|-------------|---------------|-----------------|
| TinyHAR | 249.01 | 107.48 | 62.70 | 13.51 | 24 | 19.14 |
| MLPHAR | 114.81 | 342.41 | 40.54 | 11.99 | 11 | 8.73 |
| TinierHAR | 81.42 | 39.54 | 14.43 | 25.58 | 8 | 6.36 |
| **LITEWAY-F** | 56.71 | 10.63 | 16.35 | 33.86 | 5 | 4.35 |
| **LITEWAY-L** | 37.44 | 10.07 | 16.03 | 30.54 | 3 | 2.90 |


[results_main_box.pdf](https://github.com/user-attachments/files/28199876/results_main_box.pdf)
<img width="710" height="376" alt="results_main_box" src="https://github.com/user-attachments/assets/7859758e-7310-443f-a4d6-e8b7d4521682" />

