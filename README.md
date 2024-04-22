# NeuS-manhattan
The NeuS-Manhattan project leverages NeuS as its foundation, integrating the Manhattan-world assumption and a coarse-to-fine training strategy to enhance its performance specifically within the context of building scenarios.

## Usage
1. The coordinate system utilized in NeuS-Manhattan diverges from that of NeuS, necessitating absolute coordinates instead of relative ones as mandated by the Manhattan-world assumption. Consequently, we employ Agisoft Metashape for sparse reconstruction, and introduce agi2neus.py to adapt the output .xml format camera file for compatibility with NeuS-Manhattan. Additionally, the inclusion of the sparse cloud is imperative to this process.
2. implement exp_runner.py

## Notice
1. Other than object mask, NeuS-manhattan also needs semantic mask(red for horizontal plane, blue for vertical facet and green for other districts) 

## Acknowledgement

The whole project is based on NeuS(https://github.com/Totoro97/NeuS). Some code snippets are borrowed from [Manhattan-SDF](https://github.com/zju3dv/manhattan_sdf) and [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF). Thanks for these great projects.
