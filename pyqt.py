import sys
import queue
import pyaudio
import sounddevice as sd
import soundfile as sf
import wave
from PyQt5 import QtWidgets, QtGui
from emotion_recognition import EmotionRecognizer
import sounddevice as sd
import pyaudio
from tkinter import *
import queue
import os

from sys import byteorder
from array import array
from struct import pack

import threading
from tkinter import messagebox
from utils import get_best_estimators




class VoiceRecorderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.recording = False
        self.file_exists = False
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 720, 400)
        self.setWindowTitle("Violence & Emotion Detection")
        self.setStyleSheet("background-color: #107dc2;")

        self.q = queue.Queue()

        # Initialize the emotion recognizer
        estimators = get_best_estimators(True)
        _, estimator_dict = get_estimators_name(estimators)
        features = ["mfcc", "chroma", "mel"]
        self.detector = EmotionRecognizer(estimator_dict["BaggingClassifier"], emotions=["sad", "neutral", "happy"],
                                          features=features, verbose=0)
        self.detector.train()

        # Title Label
        title_label = QtWidgets.QLabel("Voice Recorder", self)
        title_label.setGeometry(0, 0, 720, 50)
        title_label.setStyleSheet("background-color: #107dc2; color: white; font-size: 24px;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)

        # Buttons
        record_button = QtWidgets.QPushButton("Record Audio", self)
        record_button.setGeometry(220, 100, 120, 40)
        record_button.setStyleSheet("background-color: #107dc2; color: white; font-size: 16px;")
        record_button.clicked.connect(self.record_audio)

        stop_button = QtWidgets.QPushButton("Stop Recording", self)
        stop_button.setGeometry(220, 160, 120, 40)
        stop_button.setStyleSheet("background-color: #107dc2; color: white; font-size: 16px;")
        stop_button.clicked.connect(self.stop_recording)
        stop_button.setEnabled(False)

        play_button = QtWidgets.QPushButton("Play Recording", self)
        play_button.setGeometry(220, 220, 120, 40)
        play_button.setStyleSheet("background-color: #107dc2; color: white; font-size: 16px;")
        play_button.clicked.connect(self.play_recording)
        play_button.setEnabled(False)

    def record_audio(self):
        self.recording = True
        self.file_exists = False
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.play_button.setEnabled(False)
        self.threading_rec(1)

    def stop_recording(self):
        self.recording = False
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.play_button.setEnabled(True)
        self.threading_rec(2)

    def play_recording(self):
        if self.file_exists:
            data, fs = sf.read("trial.wav", dtype='float32')
            sd.play(data, fs)
            sd.wait()
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Record something to play")

    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def threading_rec(self, x):
        if x == 1:
            filename = "trial.wav"
            threading.Thread(target=self.record_to_file, args=(filename,)).start()
            result = self.detector.predict(filename)
            QtWidgets.QMessageBox.information(self, "Emotion Recognition Result", f"The predicted emotion is: {result}")

        elif x == 2:
            filename = "trial.wav"
            threading.Thread(target=self.display, args=(filename,)).start()

    def display(self, filename):
        result = self.detector.predict(filename)
        QtWidgets.QMessageBox.information(self, "Emotion Recognition Result", f"The predicted emotion is: {result}")

    def is_silent(self, snd_data):
        return max(snd_data) < 500

    def normalize(self, snd_data):
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i * times))
        return r

    def trim(self, snd_data):
        def _trim(snd_data):
            snd_started = False
            r = array('h')

            for i in snd_data:
                if not snd_started and abs(i) > 500:
                    snd_started = True
                    r.append(i)

                elif snd_started:
                    r.append(i)
            return r

        snd_data = _trim(snd_data)

        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def add_silence(self, snd_data, seconds):
        r = array('h', [0 for i in range(int(seconds * 16000))])
        r.extend(snd_data)
        r.extend([0 for i in range(int(seconds * 16000))])
        return r

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, output=True,
                        frames_per_buffer=1024)

        num_silent = 0
        snd_started = False
        r = array('h')

        while self.recording:
            snd_data = array('h', stream.read(1024))

            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

            silent = self.is_silent(snd_data)

            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > 30:
                break

        sample_width = p.get_sample_size(pyaudio.paInt16)
        stream.stop_stream()
        stream.close()
        p.terminate()

        r = self.normalize(r)
        r = self.trim(r)
        r = self.add_silence(r, 0.5)
        return sample_width, r

    def record_to_file(self, path):
        sample_width, data = self.record()
        data = pack('<' + ('h' * len(data)), *data)

        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(16000)
        wf.writeframes(data)
        wf.close()


def get_estimators_name(estimators):
    result = ['"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in
                               zip(result, estimators)}


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VoiceRecorderApp()
    window.show()
    sys.exit(app.exec_())
