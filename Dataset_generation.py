import numpy as np

def generate_circle_image(size=128):
    image = np.zeros((size, size, 3), dtype=np.float32)
    mask = np.zeros((size, size, 1), dtype=np.float32)

    # random circle
    cx, cy = np.random.randint(30, 98), np.random.randint(30, 98)
    r = np.random.randint(10, 25)

    y, x = np.ogrid[:size, :size]
    dist = (x - cx)**2 + (y - cy)**2 <= r**2

    image[dist] = np.random.uniform(0.5, 1.0)   # random color
    mask[dist] = 1.0                            # segmentation mask

    return image, mask


def load_dataset():
    X = []
    Y = []
    for _ in range(800):   # total samples
        img, mask = generate_circle_image()
        X.append(img)
        Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)

    # train/test split
    X_train, Y_train = X[:600], Y[:600]
    X_test, Y_test = X[600:], Y[600:]

    return X_train, Y_train, X_test, Y_test
