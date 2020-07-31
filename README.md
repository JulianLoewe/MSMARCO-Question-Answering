# Setup Instructions

Start Requirements: Python 3.8 Installation with Jupyter Notebook (jupyterlab) and Pytorch installed.

To use the given Pytorch Lightning Wrapper Jupyter Notebook you have to do the following:

1. Create a Folder "DATA" in the root directory of the project.
2. Download the dataset ('train_v2.1.json','dev_v2.1.json') and put it in the created folder (https://microsoft.github.io/msmarco/)
3. Download the pretrained word embeddings ('glove.840B.300d.txt') and put it in the created folder (https://nlp.stanford.edu/projects/glove/)
4. Run 'python jupyter lab PytorchLightningWrapper.ipynb' and execute the cells accordingly.

# Notes
- There are some changes in the BiDAF Model files to get our Wrapper to work.
