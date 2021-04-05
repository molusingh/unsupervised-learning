# cs7641-unsupervised-learning

# Conda Setup Instructions (need conda, can get miniconda off chocolatey for windows or homebrew on mac)
### Using conda to create python environment
conda env create -f environment.yml

### activate the environemnt
conda activate cs7641

### if needed, add debugger
jupyter labextension install @jupyterlab/debugger

### update environment after changes to environment.yml file (deactivate env first)
conda env update --file environment.yml --prune

### Open up jupyter lab to access notebook if desired
jupyter lab

# generate final results, outputs charts in ./output directory for first dataset
python main.py --exp1 --exp2 --exp3 --exp4 --exp5
# generate final results, outputs charts in ./output directory for second dataset
python main.py --exp1 --exp2 --exp3 --eye

References:

K Means Clustering:
https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f

PCA:
https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0

Dataset Sources
diabetes: https://www.kaggle.com/uciml/pima-indians-diabetes-database
Eye Data: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State

