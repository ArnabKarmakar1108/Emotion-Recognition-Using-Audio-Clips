from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC

my_model = SVC()

rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
# train the model
rec.train()

# check the test accuracy for that model
print("Test score:", rec.test_score())

# check the train accuracy for that model
print("Train score:", rec.train_score())

# this is a neutral speech from emo-db from the testing set
print("Prediction:", rec.predict("data/emodb/wav/15a04Nc.wav"))

# this is a sad speech from TESS from the testing set
print("Prediction:", rec.predict("data/validation/Actor_25/25_01_01_01_back_sad.wav"))