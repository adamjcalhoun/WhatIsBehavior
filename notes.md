## Creating conda environment
conda create -n wib python=3
conda activate wib
conda install matplotlib numpy h5py seaborn scikit-learn
conda install -c conda-forge umap-learn
(note that conda installing umap-learn didn't work and I had to install with pip)
pip install prince