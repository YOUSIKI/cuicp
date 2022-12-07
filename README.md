# Iterative closest point algorithm accelerated by CUDA

## Environment

Major dependencies:

- `cupy` for GPU accelerated linear algebra.
- `cuml` for GPU accelerated nearest neighbor.
- `tqdm` for progress bar.

### Conda

```bash
conda create -n cuicp -c rapidsai -c conda-forge -c nvidia  \
    cuml=22.10 python=3.9 cudatoolkit=11.5 cupy=11.3 tqdm

conda activate cuicp
```

### Docker

```bash
docker build -t cuicp -f Dockerfile .

docker run --rm -it \
    --shm-size=1g \
    --ulimit memlock=-1 \
    -v $(pwd):/workspace \
    cuicp \
    python ...
```

## Running

```bash
python icp.py \
    --source /path/to/source.xyz \
    --target /path/to/target.xyz \
    --output /path/to/output.xyz
```

For detailed usage, please refer to `icp.py`.

## Example

The example data can be found [here](https://github.com/YOUSIKI/cuicp/releases/download/cuicp/example_data.tar.gz). Please download the tar and extract to `./data/cal`. Use the script `scripts/run_icp.sh` to align the example point clouds placed in `data/cal` and save to `data/out`. Then visualize the original and aligned point clouds in MeshLab (the original point clouds on the left and the aligned ones on the right). Each iteration takes about 0.28s on NVIDIA RTX3090 (with Intel i9-12900K).

![MeshLab visualization](static/meshlab_visualization.png)

## License

MIT license.
