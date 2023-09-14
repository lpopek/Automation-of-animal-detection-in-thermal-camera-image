import matplotlib as plt

def plot_img(img, title='default', grey=True, legend=None):
    if grey is True:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title)
    plt.show()