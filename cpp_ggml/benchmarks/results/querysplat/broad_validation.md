# Broad official-split validation (cpp f32 + f32-accuracy vs official py)

cases: 14 run / 14 attempted locally (2 of 55 official scenes on disk); collapses: 0; alignment gaps (cpp-vs-py < 40 dB): 0

Per-case comparison figures: [charts/querysplat/broad/](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad)

| case | views | mode | cpp-vs-GT | min cpp-vs-GT | py-vs-GT | cpp-vs-py | verdict | figure |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 2views | 25.44 | 20.32 | 25.44 | 65.94 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_000.png) |
| 1 | 2 | 2views | 28.22 | 28.11 | 28.22 | 64.33 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_001.png) |
| 2 | 2 | 2views | 30.33 | 30.28 | 30.33 | 64.51 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_002.png) |
| 3 | 2 | 2views | 31.16 | 31.05 | 31.16 | 64.71 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_003.png) |
| 4 | 4 | 4views | 27.53 | 25.86 | 27.55 | 53.0 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_004.png) |
| 5 | 4 | 4views | 25.59 | 22.41 | 25.6 | 51.7 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_005.png) |
| 6 | 4 | 4views | 28.53 | 27.32 | 28.54 | 53.55 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_006.png) |
| 7 | 4 | 4views | 29.11 | 28.45 | 29.12 | 54.22 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_007.png) |
| 8 | 4 | 4views | 25.88 | 25.12 | 25.88 | 51.47 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_008.png) |
| 9 | 4 | 4views | 29.28 | 27.42 | 29.29 | 54.17 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_009.png) |
| 10 | 4 | 4views | 28.25 | 27.4 | 28.28 | 51.01 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_010.png) |
| 11 | 4 | 4views | 26.09 | 25.18 | 26.09 | 51.01 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_011.png) |
| 12 | 4 | 4views | 28.05 | 23.55 | 28.05 | 54.01 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_012.png) |
| 13 | 4 | 4views | 27.85 | 26.56 | 27.85 | 54.4 | OK | [fig](https://github.com/Asher-1/querysplat-ggml/blob/main/cpp_ggml/benchmarks/charts/querysplat/broad/case_013.png) |

**means**: cpp-vs-GT 27.95 dB, py-vs-GT 27.96 dB, cpp-vs-py 56.29 dB
