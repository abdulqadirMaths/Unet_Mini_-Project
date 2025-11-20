from dataset_generation import load_dataset
from unet_model import unet_model

# Load dataset
X_train, Y_train, X_test, Y_test = load_dataset()

# Build UNet model
model = unet_model()

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=5,
    batch_size=8
)

# Optional: save the model
model.save("unet_model.h5")
print("Model training complete and saved as unet_model.h5")
