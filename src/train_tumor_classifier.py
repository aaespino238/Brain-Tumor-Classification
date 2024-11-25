from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from models.cnn import get_model
from utils.dataset import get_class_paths

# parameters
batch_size = 8
num_epochs = 5
image_size = (299,299)
input_shape = image_size + (3,)
validation_split = 0.2
num_classes = 4
patience = 50
base_path = '../trained_models/'
model_name = 'XCEPTION'

# image generator
image_generator = ImageDataGenerator(
    rescale=1/255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True
)
test_generator = ImageDataGenerator(rescale=1/255)

# model parameters/compilation
model_parameters = {
    "input_shape": input_shape,
    "num_classes": num_classes
}
model = get_model(model_name, model_parameters)

# callbacks

# interesting
###################### MINI XCEPTION ########################################################
# trained_models_path = base_path + model_name
# log_file_path = trained_models_path + '.log'
# csv_logger = CSVLogger(log_file_path, append=False)
# early_stop = EarlyStopping('val_loss', patience=patience)
# reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
# model_path = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.h5'
# model_checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True)
# callbacks = [model_checkpoint, early_stop, reduce_lr, csv_logger]
##############################################################################################


############################ XCEPTION ########################################################
trained_models_path = base_path + model_name
model_path = trained_models_path + '.h5'
log_file_path = trained_models_path + '.log'
csv_logger = CSVLogger(log_file_path, append=False)
callbacks = [csv_logger]
##############################################################################################

# loading dataset
df = get_class_paths()
train_df, test_df = train_test_split(df, test_size=validation_split, stratify=df['Class'])
valid_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['Class'])

train_generator = image_generator.flow_from_dataframe(train_df, x_col='Class Path', y_col='Class', batch_size=batch_size, target_size=image_size)
valid_generator = image_generator.flow_from_dataframe(valid_df, x_col='Class Path', y_col='Class', batch_size=batch_size, target_size=image_size)
test_generator = test_generator.flow_from_dataframe(test_df,  x_col='Class Path', y_col='Class', batch_size=16, target_size=image_size, shuffle=False)

model.fit(train_generator, epochs=num_epochs, validation_data=valid_generator, callbacks=callbacks, )
model.save(model_path)