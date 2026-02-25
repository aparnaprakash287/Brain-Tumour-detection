import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

DATASET_PATH=r"C:\Users\USER\Desktop\APARNA\aparna project brain tumour\data image"

#image data generators
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1 #10% of training data for validation
)
#Training data
train_generator=train_datagen.flow_from_directory(
    DATASET_PATH + "\\Training",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
#validation data
val_generator=train_datagen.flow_from_directory(
    DATASET_PATH + "\\Training",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
#Test data
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(
    DATASET_PATH + "\\Testing",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical"
)
#CNN Model
model=Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(512,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512,activation="relu"),
    Dropout(0.5),
    Dense(4,activation="softmax")#4 classes:Glioma,Meningioma,Pituitary,No Tumour
])
#Complile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_generator,validation_data=val_generator, epochs=20)

img_path=r"C:\Users\USER\Desktop\APARNA\aparna project brain tumour\data image\meningloma.jpg"

class_labels=["Glioma","Meningioma","No Tumour","Pituitary"]

#Load and preprocess image
img=load_img(img_path,target_size=(150,150))
img_array=img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=img_array/255.0

prediction=model.predict(img_array)
predicted_index=np.argmax(prediction,axis=1)[0]
predicted_label=class_labels[predicted_index]
confidence=np.max(prediction)*100
print(f"\nPredicted Class:{predicted_label}")
print(f"Confidence:{confidence:.2f}%")
plt.imshow(img)
plt.title(f"Prediction:{predicted_label} ({confidence:.2f}%)")
plt.axis("off")
plt.show()