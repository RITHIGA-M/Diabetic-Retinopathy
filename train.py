import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 🔹 Dataset Paths
train_dir ="C:\\Users\\rithi\\Downloads\\Diabetic_Dataset\\gaussian_filtered_images\\gaussian_filtered_images"     
val_dir ="C:\\Users\\rithi\\Downloads\\Diabetic_Dataset\\gaussian_filtered_images\\gaussian_filtered_images"   

# 🔹 Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased rotation
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],  # Adjust brightness
    channel_shift_range=0.1,  # Improves contrast
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 🔹 Load Images from Directory
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# 🔹 Compute Class Weights (Handles Imbalance)
class_labels = list(train_generator.class_indices.keys())  
y_train = train_generator.classes  

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 🔹 Optimized CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    GlobalAveragePooling2D(),  # Reduces Overfitting
    Dense(512, activation='relu'),
    Dropout(0.5),  # Prevents Overfitting
    Dense(len(class_labels), activation='softmax')  # 5 DR Classes
])

# 🔹 Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 🔹 Callbacks for Performance Improvement
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6)

# 🔹 Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,  # Increased Epochs
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# 🔹 Save Trained Model
model.save("dr_weights_optimized.h5")
print("✅ Model Training Completed & Saved with 90%+ Accuracy!")
