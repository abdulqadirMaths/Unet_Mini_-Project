import matplotlib.pyplot as plt

# Predict on first 5 test images
preds = model.predict(X_test[:5])

plt.figure(figsize=(12, 6))
for i in range(5):
    # Original image
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i])
    plt.title("Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(3, 5, i + 6)
    plt.imshow(Y_test[i].squeeze(), cmap='gray')
    plt.title("True Mask")
    plt.axis("off")

    # Predicted mask
    plt.subplot(3, 5, i + 11)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

plt.show()
