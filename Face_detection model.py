#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations')


# In[129]:


import os
import time
import uuid
import cv2


# In[130]:


import os
print(os.getcwd())


# In[131]:


import os
os.makedirs('images', exist_ok=True)


# In[19]:


IMAGES_PATH = os.path.join('data','images')
number_images = 30


# In[186]:


cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[187]:


get_ipython().system('labelme')


# In[13]:


get_ipython().system('pip install tensorflow')


# In[188]:


import tensorflow as tf
import json #labels are in json format 
import numpy as np #help with data preprocessing 
from matplotlib import pyplot as plt #visualize our images 


# In[189]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[190]:


tf.config.list_physical_devices('GPU')


# In[200]:


images = tf.data.Dataset.list_files('data\\images\\*.jpg')


# In[201]:


images.as_numpy_iterator().next()


# In[202]:


def load_image(x): #load an image file using tensorflow 
    byte_img = tf.io.read_file(x) #reads the file as a sequence of bytes 
    img = tf.io.decode_jpeg(byte_img) #converts the byte-encoded image data into a TensorFlow tensor representing the image and assigns it to the img variable
    return img


# In[203]:


images = images.map(load_image) #using the map() function in TensorFlow to apply the load_image function to each element in the images dataset


# In[204]:


images.as_numpy_iterator().next()


# In[205]:


type(images) #viewing tensorflow data pipeline in output 


# In[206]:


image_generator = images.batch(4).as_numpy_iterator()


# In[207]:


plot_images = image_generator.next() #going to return a new batch of data each time 


# In[208]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image) 
plt.show()


# In[209]:


for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels', filename)
        if os.path.exists(existing_filepath): 
            new_filepath = os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath, new_filepath) 


# In[27]:


get_ipython().system('pip install albumentations')


# In[210]:


import albumentations as alb #to perform image augmentation on images and labels 


# In[211]:


augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), #horizontally flips the input image with a probability of 0.5 (50% chance).
                         alb.RandomBrightnessContrast(p=0.2), 
                         alb.RandomGamma(p=0.2), #randomly adjusts the gamma (nonlinear brightness) of the input image
                         alb.RGBShift(p=0.2), # randomly shifts the RGB channels of the input image
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', #specifies the parameters for bounding box annotations. It sets the format to 'albumentations' and specifies the label field name as 'class_labels'. 
                                                  label_fields=['class_labels']))


# In[212]:


img = cv2.imread(os.path.join('data','train', 'images','0f3a7763-1e2f-11ee-802c-ff658ee17ec9.jpg'))


# In[213]:


img


# In[214]:


with open(os.path.join('data', 'train', 'labels', '0f3a7763-1e2f-11ee-802c-ff658ee17ec9.json'), 'r') as f:
    label = json.load(f)


# In[215]:


label


# In[216]:


label['shapes'][0]['points']


# In[217]:


coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]


# In[218]:


coords


# In[219]:


coords = list(np.divide(coords, [640,480,640,480])) #dividing our coordinates by width and height of images


# In[220]:


coords


# In[221]:


augmented = augmentor(image=img, bboxes=[coords], class_labels=['face']) #calling the augmentor function written earlier


# In[222]:


augmented['bboxes']


# In[223]:


cv2.rectangle(augmented['image'], 
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)), #represent the first 2 values
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), #represent the last two values 
                    (255,0,0), 2) #specifying the colour, 2 is the thickness of the actual image 

plt.imshow(augmented['image'])


# In[224]:


for partition in ['train','test','val']: 
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001] #coordinates for images that dont have labels 
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json') 
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))

        try: 
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: #takes the first bounding box from the augmented image and assigns it to the annotation's bounding box, while setting the class label to 1.
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)      


# In[225]:


train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False) #loading augmented images to tensorflow dataset
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)


# In[226]:


test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255) #getting image size between 0 and 1 so we can use the sigmoid function


# In[227]:


val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)


# In[228]:


train_images.as_numpy_iterator().next()


# In[229]:


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']


# In[230]:


# now loading labels to tensorflow dataset for the json file 
# in the second statement, we are going through each label-x to get their class and bbox 
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# In[231]:


test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# In[232]:


val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# In[233]:


train_labels.as_numpy_iterator().next()


# In[234]:


# check partition lenghts
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)


# In[235]:


train = tf.data.Dataset.zip((train_images, train_labels)) # combine the training images and labels 
train = train.shuffle(5000) 
train = train.batch(8) # each batch will be represented as 8 images and 8 labels 
train = train.prefetch(4) # model can perform computations on the current batch while the next batch is being prepared 


# In[236]:


test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)


# In[237]:


val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)


# In[238]:


train.as_numpy_iterator().next()[1] # in the output we will get all our classes and bounding boxes 


# In[239]:


data_samples = train.as_numpy_iterator() # allows us to loop through all of the different batches 


# In[240]:


res = data_samples.next() # to grab the next batch


# In[241]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = res[0][idx] #retrieves the image corresponding to the current index 
    sample_coords = res[1][1][idx] #retrives the coordinates of the current index
    # rectangle is defined by two parts, top left and bottom right coordinates 
    cv2.rectangle(sample_image, 
                  tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                        (255,0,0), 2) #represents color of rectangle and thickness of rectangle border 

    ax[idx].imshow(sample_image)


# In[242]:


from tensorflow.keras.models import Model # base model class 
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D # different layers 
from tensorflow.keras.applications import VGG16 # huge neural network that is pre-trained, classification model 


# In[243]:


vgg = VGG16(include_top=False) # gets rid of the final layers that we dont require 


# In[244]:


vgg.summary() #none in the output represents number of samples, width+height, number of channels  


# In[245]:


def build_model(): 
    input_layer = Input(shape=(120,120,3))
    # passing input layer to vgg layer 
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model  
    f1 = GlobalMaxPooling2D()(vgg) # condensing all info from vgg layer using global max pooling layer, will return only the max values 
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    # classification output and regression output 
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


# In[246]:


facetracker = build_model()


# In[247]:


facetracker.summary()


# In[248]:


X, y = train.as_numpy_iterator().next() # x-images, y-labels 


# In[249]:


X.shape


# In[250]:


classes, coords = facetracker.predict(X)


# In[251]:


classes, coords


# In[252]:


batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch # decay factor per batch 


# In[253]:


opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)
# creates an instance of the adam optimizer 
# learning rate controls how quickly the model learns from the gradient updates 
# Decay is applied to the learning rate over time to gradually reduce it and improve convergence


# In[254]:


def localization_loss(y_true, yhat):   # calculates the localization loss between the predicted bounding boxes (yhat) and the true bounding boxes (y_true).         
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2])) # calculates the coordinate loss component of the localization loss
    # calculate the height (h_true) and width (w_true) of the true bounding boxes by subtracting the x and y coordinates accordingly.             
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 
    # same for predicted bounding boxes 
    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    # calculates the size loss component of the localization loss
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    # to quantify the discrepancy between the predicted and true bounding boxes.
    return delta_coord + delta_size


# In[255]:


classloss = tf.keras.losses.BinaryCrossentropy() #passed to keras pipeline
regressloss = localization_loss


# In[256]:


localization_loss(y[1], coords)


# In[257]:


classloss(y[0], classes)


# In[258]:


regressloss(y[1], coords)


# In[259]:


class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): # eyetracker is a pre-built neural network 
        super().__init__(**kwargs) # **kwargs notation allows for any additional keyword arguments to be passed to the constructor
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt # The optimizer will be used for updating the model's parameters during training
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: # to trace the operations performed during the forward pass
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes) # calculates the class loss using the class loss function (self.closs) by comparing the predicted classes (classes) with the target class labels (y[0])
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords) # comparing the predicted coordinates (coords) with the target bounding box coordinates (y[1])
            
            total_loss = batch_localizationloss+0.5*batch_classloss 
            
            grad = tape.gradient(total_loss, self.model.trainable_variables) # calculates the gradients of the total_loss with respect to the trainable variables of the model
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables)) # applies the computed gradients to update the model's trainable variables using the optimizer (opt). It performs the gradient descent step
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): # same step without the gradient calculations 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)


# In[260]:


model = FaceTracker(facetracker)


# In[261]:


model.compile(opt, classloss, regressloss)


# In[262]:


logdir='logs'


# In[263]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[264]:


hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])


# In[265]:


hist.history


# In[266]:


fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()


# In[267]:


test_data = test.as_numpy_iterator()


# In[268]:


test_sample = test_data.next()


# In[269]:


yhat = facetracker.predict(test_sample[0])


# In[270]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]
    
    if yhat[0][idx] > 0.9: #checking if classification loss is greater than 0.9 
        cv2.rectangle(sample_image, 
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)
    
    ax[idx].imshow(sample_image)


# In[271]:


from tensorflow.keras.models import load_model # saving the model 


# In[272]:


facetracker.save('facetracker.h5') # saving the model 


# In[273]:


facetracker = load_model('facetracker.h5')


# In[274]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




