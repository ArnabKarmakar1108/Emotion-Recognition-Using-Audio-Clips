#Import necessary modules
from emotion_recognition import EmotionRecognizer
import sounddevice as sd
import simpleaudio as sa
import soundfile as sf
import numpy as np
import pyaudio
from tkinter import *
import queue
import os
import wave
from pydub import AudioSegment
from sys import byteorder
from array import array
from struct import pack
import threading
from tkinter import Toplevel, Label
from tkinter import messagebox
from utils import get_best_estimators

#Define the user interface
voice_rec = Tk()
voice_rec.geometry("1400x600")
voice_rec.title("Violence & Emotion Detection")
voice_rec.config(bg="#272727")

#Create a queue to contain the audio data
q = queue.Queue()
#Declare variables and initialise them
recording = False
file_exists = True 

#Fit data into queue
def callback(indata, frames, time, status):
    q.put(indata.copy())

#Functions to play, stop and record audio
#The recording is done as a thread to prevent it being the main process
def threading_rec(x):
    global recording
    if x == 1:
        # If recording is selected, then the thread is activated
        filename = "trial.wav"
        t1 = threading.Thread(target=record_to_file, args=(filename,))
        t1.start()
        recording = True

    elif x == 2:
        # To stop, set the flag to false
        recording = False
        # Note: We removed the prediction functionality from the stop button

    elif x == 3:
        # To play a recording, check if the file exists
        filename = "trial.wav"
        if not os.path.exists(filename):
            messagebox.showerror(message="No file found. Record something to play.")
            return

        # Read the recording and play it using pydub
        audio = AudioSegment.from_file(filename)
        audio.export("temp.wav", format="wav")
        os.system("start temp.wav")

    elif x == 4:  # New button for predicting
        filename = "trial.wav"
        if not os.path.exists(filename):
            messagebox.showerror(message="No file found. Record something to predict.")
            return

        t1 = threading.Thread(target=display_result, args=(filename,))
        t1.start()


def display_result(filename):
    result = detector.predict(filename)

    # Create a custom styled toplevel window
    custom_msg_box = Toplevel(voice_rec)
    custom_msg_box.title("Result")
    custom_msg_box.geometry("300x150")
    custom_msg_box.config(bg="#fbca36")

    # Create and style the message label
    message_label = Label(custom_msg_box, text=result, font=("Arial", 20), bg="#fbca36", padx=20, pady=20)
    message_label.pack()

    # Add an OK button to close the custom messagebox
    ok_button = Button(custom_msg_box, text="OK", font=("Arial", 12), bg="#272727", fg="#fbca36", command=custom_msg_box.destroy)
    ok_button.pack()

    # Since the messagebox is created in the display_result function, we need to call mainloop() for it to work correctly.
    voice_rec.mainloop()

def delete_file():
    filename = "trial.wav"
    if os.path.exists(filename):
        os.remove(filename)
        messagebox.showinfo(message=f"{filename} deleted successfully.")
    else:
        messagebox.showerror(message=f"{filename} does not exist.")

def terminate_window():
    voice_rec.destroy()

THRESHOLD = 500     #16 - bit signed integer below which audio will be considered silent
CHUNK_SIZE = 1024   #No of frames per chunk
FORMAT = pyaudio.paInt16
RATE = 16000        #(Hz) No of samples per second

SILENCE = 30        #No of consecutive silent chunks indicating end of recording

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r
def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    messagebox.showinfo(message= "Please Speak In The Mic")
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    import argparse
    parser = argparse.ArgumentParser(description="""
                                    Testing emotion recognition system using your voice, 
                                    please consider changing the model and/or parameters as you wish.
                                    """)
    parser.add_argument("-e", "--emotions", help=
                                            """Emotions to recognize separated by a comma ',', available emotions are
                                            "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy"
                                            """, default="sad,neutral,happy")
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are: {},
                                        default is "BaggingClassifier"
                                        """.format(estimators_str), default="BaggingClassifier")


    # Parse the arguments passed
    args = parser.parse_args()

    features = ["mfcc", "chroma", "mel"]
    detector = EmotionRecognizer(estimator_dict[args.model], emotions=args.emotions.split(","), features=features, verbose=0)
    detector.train()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))
        
    # Label to display app title
    title_lbl = Label(voice_rec, text="Voice Recorder & Prediction", font=("Arial", 24), bg="#fbca36")
    title_lbl.grid(row=0, column=0, columnspan=7, pady=(20, 10))

     # Button to record audio
    record_btn = Button(voice_rec, text="Record Audio", font=("Arial", 12), bg="#fbca36", fg="#272727",
                        command=lambda m=1: threading_rec(m))
    record_btn.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

    # Stop button
    stop_btn = Button(voice_rec, text="Stop Recording", font=("Arial", 12), bg="#fbca36", fg="#272727",
                      command=lambda m=2: threading_rec(m))
    stop_btn.grid(row=1, column=3, padx=10, pady=10, sticky="nsew")

    # Button to predict
    predict_btn = Button(voice_rec, text="Predict Emotion", font=("Arial", 12), bg="#fbca36", fg="#272727",
                         command=lambda m=4: threading_rec(m))
    predict_btn.grid(row=1, column=4, padx=10, pady=10, sticky="nsew")

    # Play button
    play_btn = Button(voice_rec, text="Play Recording", font=("Arial", 12), bg="#fbca36", fg="#272727",
                      command=lambda m=3: threading_rec(m))
    play_btn.grid(row=1, column=5, padx=10, pady=10, sticky="nsew")

    # Button to delete the file
    delete_btn = Button(voice_rec, text="Delete File", font=("Arial", 12), bg="#fbca36", fg="#272727",
                        command=delete_file)
    delete_btn.grid(row=1, column=6, padx=10, pady=10, sticky="nsew")

    # Button to terminate the window
    terminate_btn = Button(voice_rec, text="Terminate", font=("Arial", 12), bg="#fbca36", fg="#272727",
                           command=terminate_window)
    terminate_btn.grid(row=1, column=7, padx=10, pady=10, sticky="nsew")



    # Add a label to display instructions
    instructions_lbl = Label(voice_rec, text="Click 'Record Audio' to start recording, 'Stop Recording' to stop, 'Play Recording' to play and 'Delete File' to delete the file", font=("Arial", 10), bg="#272727", fg="#ffffff")
    instructions_lbl.grid(row=2, column=2, columnspan=3, pady=(0, 20))

    voice_rec.mainloop()

