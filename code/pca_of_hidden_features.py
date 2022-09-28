import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

#%%
#x = np.random.normal(3, 2.5, size=(5, 4))
#b_labels = np.array([[0,1], [1,0], [0,1], [1,0], [0,1]])

#pca_of_hidden_features(x, b_labels)
def pca_of_hidden_features(x, b_labels):
    """
    Output representation of the posts in 2D space with labels
    to understand the separation between the two. RED means nta,
    and BLUE means yta
    :param x: input feature matrix
    :param b_labels: the labels of the points in space
    :return: saves plot
    """
    pca = PCA(n_components=2)
    x = pca.fit_transform(x)

    # labels in 0 and 1
    labels = b_labels[:,0]
    colormap = np.array(['r', 'b'])

    plt.scatter(x[:, 0], x[:, 1], label=labels, c=colormap[labels])
    plt.title('PCA of the lower representations of hidden features of the posts')
    red_patch = mpatches.Patch(color='red', label='YTA')
    blue_patch = mpatches.Patch(color='blue', label='NTA')

    plt.legend(handles=[red_patch, blue_patch])
    plt.savefig('results/PCA_representation.jpg')

    plt.show()
