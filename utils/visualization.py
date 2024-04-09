import matplotlib.pyplot as plt


def show_batch(batch, nrows=4, ncols=4, figsize=2):
    assert len(batch['label']) == ncols * nrows

    figure, axs = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))

    for n, (img, label) in enumerate(zip(batch['image'], batch['target'])):
        i, j = n // ncols, n % ncols
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].set_title(f'class - {label}')
        axs[i, j].set_axis_off()

    plt.show()


if __name__ == '__main__':
    import pandas as pd

    m = pd.read_csv('../train_balanced_acc.csv', header=None)

    plt.plot(m[0], m[2])
    plt.show()
