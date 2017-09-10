from keras.models import load_model


model = load_model("C:/Users/domin/Documents/GitHub/trained_models/SiamCNN_Model_05_09_2017_00-26-33.h5")

DATA_DIR = "../"
# IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
IMAGE_DIR = os.path.join(DATA_DIR, "yalefaces")


model.predict(X,batch_size=16)

print("Model loaded...")