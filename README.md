# Synthesizing Interpretable Control Policies
This repo is associated to the paper [Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis](https://arxiv.org/abs/2510.00373).
There is a wandb setup to have an intuitive logging of what is going on (best scores achieved, best islands...). This can be disregarded by selecting the option to not track the experiment from command line.
We provide a Docker container to make it easier to reproduce results.
The `dm_control_tests/` folder contains scripts to visualize what the different tasks are doing when in closed-loop with a policy. There are many tasks available, we have not tried them all.

## Docker
Build hierarchical docker image, to avoid installing everything over and over again.

#### Build the base image
```
docker build -t funsearch-base:latest -f Dockerfile.base .
docker build . -t funsearch:latest
```

#### Run the docker image
```
# When running on server (share huggingface cache so you don't need to download the model weights every time):
> docker build . -t funsearch:latest && docker run --gpus all -it -v $(pwd)/data:/workspace/data -v /home/$USER/.cache/huggingface:/root/.cache/huggingface funsearch

# When running on personal machine, so you can use a small model.
docker run --gpus all -it -v $(pwd):/workspace/data -v /home/$USER/ssd_data/_huggingface_cache:/root/.cache/huggingface --hostname $USER funsearch:latest
```
#### Run an experiment
All the spec files that use gradient-free optimization in the loop are in the `examples_ng/` folder.
```
> funsearch run examples_ng/dm_control_swingup_spec.py 1 --sandbox_type ExternalProcessSandbox
> funsearch run examples_ng/dm_control_ballcup_spec.py 3 --sandbox_type ExternalProcessSandbox
```
There are many flags to add when launching an experiment, the `--parametric_program` flag activates or deactivates the program optimization at evaluation time.

#### Other useful commands

```
docker images # list the images
docker rmi funsearch # remove the image
```

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@article{bosio2025combining,
  title={Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis},
  author={Bosio, Carlo and Guarrera, Matteo and Sangiovanni-Vincentelli, Alberto and Mueller, Mark W},
  journal={arXiv preprint arXiv:2510.00373},
  year={2025}
}
```
## Previous work
Check out previous works that this paper buids on top of:
[Synthesizing interpretable control policies through large language model guided search](https://ieeexplore.ieee.org/abstract/document/11107729/)
[Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6)
