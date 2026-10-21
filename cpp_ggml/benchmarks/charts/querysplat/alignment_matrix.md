# Quant x backend alignment matrix (f32-accuracy default mode)

S=4 views, 512x512, paper decoder. cpp-vs-py = render agreement
with the official torch chain run on the same inputs (higher is better;
identical 8-bit PNGs read 99). GT = vs ground truth. Seconds are one
fresh process per row (n=1, warm-up included, model load excluded).

## Official python reference (CUDA)

| scene | py-vs-GT | enc s | dec s | render s |
|---|---|---|---|---|
| case0 | 27.55 | 0.65 | 1.90 | 0.33 |
| case1 | 25.6 | 0.73 | 1.97 | 0.32 |
| courtyard | 27.41 | 0.61 | 1.92 | 0.25 |

## cpp rows

| backend | quant | scene | cpp-vs-py | cpp-vs-GT | min cpp-vs-GT | enc+dec s | render s | total s |
|---|---|---|---|---|---|---|---|---|
| cuda | f32 | case0 | 53.0 | 27.53 | 25.86 | 2.85 | 0.38 | 14.05 |
| cuda | f32 | case1 | 51.7 | 25.59 | 22.41 | 2.79 | 0.39 | 12.82 |
| cuda | f32 | courtyard | 51.5 | 27.43 | 27.05 | 2.77 | 0.42 | 14.25 |
| cuda | f16 | case0 | 47.09 | 27.53 | 25.89 | 2.76 | 0.38 | 16.1 |
| cuda | f16 | case1 | 45.32 | 25.59 | 22.43 | 2.75 | 0.38 | 12.13 |
| cuda | f16 | courtyard | 43.78 | 27.41 | 27.04 | 2.78 | 0.4 | 13.46 |
| cuda | q8_0 | case0 | 34.91 | 27.56 | 26.01 | 2.79 | 0.39 | 5.59 |
| cuda | q8_0 | case1 | 34.17 | 25.64 | 22.39 | 2.87 | 0.42 | 6.4 |
| cuda | q8_0 | courtyard | 34.44 | 27.45 | 26.71 | 2.88 | 0.41 | 7.94 |
| vulkan | f32 | case0 | 63.35 | 27.55 | 25.89 | 2.68 | 0.39 | 15.74 |
| vulkan | f32 | case1 | 64.17 | 25.6 | 22.42 | 2.71 | 0.38 | 13.6 |
| vulkan | f32 | courtyard | 62.73 | 27.41 | 27.03 | 2.61 | 0.45 | 14.36 |
| vulkan | f16 | case0 | 47.18 | 27.53 | 25.91 | 2.6 | 0.39 | 12.72 |
| vulkan | f16 | case1 | 45.71 | 25.6 | 22.44 | 2.68 | 0.4 | 12.94 |
| vulkan | f16 | courtyard | 44.04 | 27.4 | 27.03 | 2.7 | 0.41 | 12.16 |
| vulkan | q8_0 | case0 | 34.91 | 27.58 | 26.04 | 2.54 | 0.36 | 6.32 |
| vulkan | q8_0 | case1 | 34.15 | 25.65 | 22.36 | 2.48 | 0.4 | 6.18 |
| vulkan | q8_0 | courtyard | 34.52 | 27.43 | 26.7 | 2.5 | 0.42 | 7.8 |
| cpu | f32 | case0 | 62.48 | 27.55 | 25.89 | 87.91 | 0.37 | 100.23 |
| cpu | f32 | case1 | 62.77 | 25.6 | 22.42 | 80.1 | 0.43 | 90.49 |
| cpu | f32 | courtyard | 60.92 | 27.41 | 27.03 | 86.64 | 0.39 | 101.7 |
| cpu | f16 | case0 | 47.19 | 27.53 | 25.91 | 79.39 | 0.37 | 95.07 |
| cpu | f16 | case1 | 45.69 | 25.6 | 22.44 | 84.76 | 0.42 | 97.93 |
| cpu | f16 | courtyard | 44.03 | 27.41 | 27.03 | 83.91 | 0.41 | 98.34 |
| cpu | q8_0 | case0 | 34.91 | 27.58 | 26.04 | 88.61 | 0.38 | 92.85 |
| cpu | q8_0 | case1 | 34.15 | 25.65 | 22.36 | 83.48 | 0.4 | 87.68 |
| cpu | q8_0 | courtyard | 34.52 | 27.43 | 26.7 | 84.38 | 0.42 | 89.68 |

## Means per (backend, quant)

| backend | quant | mean cpp-vs-py | mean cpp-vs-GT | mean enc+dec s | mean render s |
|---|---|---|---|---|---|
| cpu | f16 | 45.64 | 26.85 | 82.69 | 0.4 |
| cpu | f32 | 62.06 | 26.85 | 84.88 | 0.4 |
| cpu | q8_0 | 34.53 | 26.89 | 85.49 | 0.4 |
| cuda | f16 | 45.4 | 26.84 | 2.76 | 0.39 |
| cuda | f32 | 52.07 | 26.85 | 2.8 | 0.4 |
| cuda | q8_0 | 34.51 | 26.88 | 2.85 | 0.41 |
| vulkan | f16 | 45.64 | 26.84 | 2.66 | 0.4 |
| vulkan | f32 | 63.42 | 26.85 | 2.67 | 0.41 |
| vulkan | q8_0 | 34.53 | 26.89 | 2.51 | 0.39 |
| official py cuda | f32 | — | 26.85 | 2.59 | 0.3 |
