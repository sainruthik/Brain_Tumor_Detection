# train.py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocessing import train_generator, test_generator
from model import build_dual_channel_model

# Hyperparameters
EPOCHS = 30
LEARNING_RATE = 0.0001

# Build and compile the model
model = build_dual_channel_model()
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Set up ModelCheckpoint to save the model based on the best validation accuracy
checkpoint = ModelCheckpoint(
    filepath="best_brain_tumor_detection_model.keras",  # Changed extension to .keras
    monitor='val_accuracy',      # Monitor validation accuracy
    verbose=1,                   # Verbose output for saving checkpoints
    save_best_only=True,         # Save only if the model's accuracy improves
    mode='max'                   # Maximize validation accuracy
)

# Training the model with ModelCheckpoint callback
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[checkpoint]       # Add checkpoint callback
)

print("Training complete. Best model saved as best_brain_tumor_detection_model.keras based on validation accuracy.")
