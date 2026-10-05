# M4 profile buckets (SparseDriveV2)

## deploy\artifacts\prof\prof_e_fix2full.log

rows=77 total_row=48.4418 sum_of_layers=48.442 ms

| bucket | ms/iter | % | layers |
|---|---:|---:|---:|
| dfa_plugin | 20.183 | 41.7% | 3 |
| myelin_fused | 22.358 | 46.2% | 4 |
| img_backbone | 4.648 | 9.6% | 52 |
| reformat | 1.044 | 2.2% | 17 |
| other | 0.210 | 0.4% | 1 |

## deploy\artifacts\prof\prof_e_fix2nt.log

rows=77 total_row=48.352 sum_of_layers=48.352 ms

| bucket | ms/iter | % | layers |
|---|---:|---:|---:|
| dfa_plugin | 20.189 | 41.8% | 3 |
| myelin_fused | 22.287 | 46.1% | 4 |
| img_backbone | 4.628 | 9.6% | 52 |
| reformat | 1.039 | 2.1% | 17 |
| other | 0.209 | 0.4% | 1 |

## deploy\artifacts\prof\prof_e_fix2f32.log

rows=61 total_row=76.9691 sum_of_layers=76.969 ms

| bucket | ms/iter | % | layers |
|---|---:|---:|---:|
| dfa_plugin | 20.018 | 26.0% | 3 |
| myelin_fused | 36.941 | 48.0% | 4 |
| img_backbone | 19.743 | 25.7% | 52 |
| reformat | 0.025 | 0.0% | 1 |
| other | 0.240 | 0.3% | 1 |
