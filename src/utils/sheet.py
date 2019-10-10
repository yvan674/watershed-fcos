"""Sheet.

Creates a visualization sheet that shows everything we need next to each other.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def process_sheet(contents):
    """Processes contents into a sheet plot for visualization.

    Args:
        contents (list): List containing dictionaries containing 'preds',
            'target', 'score', 'classes'. 'score' and 'classes' are lists with
            each element representing the output of one head.

    Returns:
        list: List of sheets as numpy ndarrays.
    """
    """Program flow:
    
    For each item in contents:
        1. Create an empty plot
        2. Fill it with the images
        3. add it to the list of output plots
    
    Return output plot 
    """
    plots = []
    nrows = 2
    ncol = 1 + len(contents[0]['score'])
    dpi = 72



    width = ((contents[0]['preds'].shape[1] * ncol) + (20 * (ncol + 1))) / dpi
    height = ((contents[0]['preds'].shape[0] * nrows)
              + (20 * (nrows + 1))) / dpi


    for content in contents:
        for i in range(2):  # Return two plots: One with blended one without
            fig = plt.figure(figsize=(width, height), dpi=dpi)
            # BBox Preds
            a = fig.add_subplot(nrows, ncol, 1)
            imgplot = plt.imshow(content['preds'])
            a.set_title('prediction')

            # BBox Targets
            a = fig.add_subplot(nrows, ncol, ncol + 1)
            imgplot = plt.imshow(content['target'])
            a.set_title('target')

            # scores
            counter = 0
            for item in content['score'][i::2]:
                a = fig.add_subplot(nrows, ncol, 2 + counter)
                imgplot = plt.imshow(item)
                a.set_title('score ' + str(counter))

                counter += 1

            counter = 0
            for item in content['classes'][i::2]:
                a = fig.add_subplot(nrows, ncol, ncol + 2 + counter)
                imgplot = plt.imshow(item)
                a.set_title('classes ' + str(counter))

                counter += 1

            fig.tight_layout()
            fig.canvas.draw()

            # Turn fig into np array and append it to the output list
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = data.transpose((2, 0, 1))
            plots.append(data)

    return plots
