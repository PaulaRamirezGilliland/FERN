import matplotlib.pyplot as plt



def plot_input_space(data, y_pred, save_path, i, overlap=False, show=False):
    """

    Args:
        data: dict containing "stacks" array (from data/scan.py), the input to the network
        y_pred: (torch tensor) predicted slice from volume (in input 2D space) - use slice_acquisition
        save_path: path to save the figure
        i: (int) casenum or index for saving
        overlap: (bool) whether to display as overlap or individual plots
        show: (bool) display plot (if False only saves)

    Returns: Plots the input space data (2D slice), given to the network and predicted slice from a volume (e.g. atlas)

    """
    if overlap:
        plt.figure(figsize=(25, 25))
        for ind in range(data['stacks'].shape[0]):
            plt.subplot(2, data['stacks'].shape[0], ind + 1)
            plt.title(f"Input 2D and Predicted Slice overlap")
            plt.imshow(data['stacks'][ind, 0, ...].detach().cpu(), cmap='gray')
            plt.imshow(y_pred[ind, 0, ...].detach().cpu(), cmap='gray',alpha=0.5)
            plt.grid(color='w', linestyle='--', linewidth=0.5)

    else:
        plt.figure(figsize=(25, 25))
        for ind in range(data['stacks'].shape[0]):
            plt.subplot(2, data['stacks'].shape[0], ind + 1)
            plt.title(f"Input 2D")
            plt.imshow(data['stacks'][ind, 0, ...].detach().cpu(), cmap='gray')
            plt.grid(color='w', linestyle='--', linewidth=0.5)

        ind_start = ind + 1
        for ind_p, subplot_index in enumerate(range(ind_start, data['stacks'].shape[0] + ind_start)):
            plt.subplot(2, data['stacks'].shape[0], subplot_index + 1)
            plt.title("Predicted Slice (from volume)")
            plt.imshow(y_pred[ind_p, 0, ...].detach().cpu(), cmap='gray')
            plt.grid(color='w', linestyle='--', linewidth=0.5)

    plt.axis('off')
    if show:
        plt.show()

    plt.tight_layout()
    plt.savefig(save_path + '/pred-in-' + str(i) + '.svg', format = 'svg', transparent=True)
    plt.savefig(save_path + '/pred-in-' + str(i) + '.png', format = 'png')

    plt.close()
    return 0

