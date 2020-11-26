##Sonify toolkit
##started by Yuma Antoine Decaux
##
##Helps with generating sound versions of black and white and RGB images, in different ways such as a directional scan, or vertically stacked left to right scan.
##Written to help with COMP3710: Pattern recognition

from collections import Counter
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.generators import Triangle
from pydub.generators import Sawtooth
import numpy as np
from scipy.interpolate import interp1d
from pandas import qcut
from os.path import isdir
from os import makedirs, listdir
from csv import writer
import pyttsx3
import time
from sonifyUtils import *
import cv2

def convertImages(dir):
	files = [f for f in listdir('./'+dir) if '.DS_' not in f and 'audio' not in f]
	for file in files:
		name = file.split('.')[0]
		print("Opening %s" % file)
		image = cv2.imread('./'+dir+'/'+file)
		image = cv2.resize(image, (64, 64))
		sonify2DColor(np.array(image), name, dir+'/audio')

def sonify2D(array, name, path, waveType=0):
	print('--Sonifying 2D array--')
	speech = speechAudio('Function is '+name+'. rows are stacked')
	speech = speech.pan(-1.0)
	size = array.shape
	ms = int(4000/size[0])
	pan = np.linspace(-1, 1, size[0])
	sigma = np.amax(array)
	normalised = (array/sigma) + 0.5
	output = sonify(normalised[0], pan, ms)
	for i in range(1, size[0], 1):
		segment = sonify(normalised[i], pan, ms)
		output = output.overlay(segment)
	speech = speech.append(output, crossfade=300)
	exportSound(name, speech, path)

def sonify2DScan(array, name, path):
	speech = speechAudio('Function is '+name+'. Scanning top left to bottom right')
	speech = speech.pan(-1.0)
	print('--Sonifying 2D array with scanning--> ', name)
	size = array.shape
	ms = int(250/size[0])
	pan = np.linspace(-1, 1, size[0])
	sigma = np.amax(array)
	normalised = (array/sigma)
	output = sonifyScan(normalised[0], pan, ms)
	rowLength = len(output)
	mid = int(rowLength/4)
	silence = AudioSegment.silent(rowLength*size[0]-mid)
	output = output.append(silence, crossfade=0)
	count = 1
	for i in range(1, size[0], 1):
		segment = sonifyScan(normalised[i], pan, ms)
		mixed = output.overlay(segment, position=mid*count)
		output = mixed
		count += 1
	print('finalised output length: ', str(len(output)))
	output = speech.append(output, crossfade=300)
	exportSound(name, output, path)

def sonify2DColor(array, name, path):
	speech = speechAudio('Function is '+name+'. Scanning top left to bottom right')
	speech = speech.pan(-1.0)
	hues = getHueScores(array)
	red = speechAudio('Red ')
	redHue = getWaveType(np.where(hues==0)[0])
	redHue = redHue - 12
	red = red.append(redHue)
	green = speechAudio('Green ')
	greenHue = getWaveType(np.where(hues==1)[0])
	greenHue = greenHue - 12
	green = green.append(greenHue)
	blue = speechAudio('Blue ')
	blueHue = getWaveType(np.where(hues==2)[0])
	blueHue = blueHue - 24
	blue = blue.append(blueHue)
	s = AudioSegment.silent(500)
	speech = speech.append(red).append(green).append(blue).append(s)
	print('--Sonifying 2D array with colors--> ', name)
	size = array.shape
	ms = int(2000/size[1])
	pan = np.linspace(-1, 1, size[1])
	sigma = np.amax(array)
	normalised = (array/(sigma+0.01))
	output = sonifyColor(normalised[0], pan, hues, ms)
	rowLength = len(output)
	mid = int(rowLength/8)
	silence = AudioSegment.silent(rowLength*size[0]-mid)
	output = output.append(silence, crossfade=0)
	count = 1
	for i in range(1, size[0], 1):
		segment = sonifyColor(normalised[i], pan, hues, ms)
		mixed = output.overlay(segment, position=mid*count)
		output = mixed
		count += 1
	print('finalised output length: ', str(len(output)))
	output = speech.append(output, crossfade=300)
	exportSound(name, output, path)

def sonify(array, pan, ms=200, factor=1.0, sampleRate=96000, bitDepth=32):
	size = len(pan)
	cNote = 83 #lowest possible e note frequency
	wave = Sine(cNote*(array[0]*factor), sample_rate=sampleRate, bit_depth=bitDepth)
	output = wave.to_audio_segment(duration=ms)
	output = output.pan(pan[0])
	for i in range(1, size, 1):
		wave = Sine(cNote*(array[i]*factor), sample_rate=sampleRate, bit_depth=bitDepth)
		segment = wave.to_audio_segment(duration=ms)
		segment = segment.pan(pan[i])
		output = output.append(segment, crossfade=5)
	return output

def sonifyScan(array, pan, ms=200, factor=1.0, sampleRate=96000, bitDepth=32):
	output = sineDot(array[0], pan[0], ms, factor, sampleRate, bitDepth)
	output = output.pan(pan[0])
	for i in range(1, len(array)):
		segment = sineDot(array[i], pan[i], ms, factor, sampleRate, bitDepth)
		silence = AudioSegment.silent(50)
		output = output.append(silence, crossfade=0)
		output = output.overlay(segment, position=0.8*len(output))
	return output

def sonifyColor(array, pan, hues, ms=200, factor=1.0, sampleRate=96000, bitDepth=32):
	factors = array[0]
	units = [int((1-x+0.01)*28) for x in factors]
	output = sineDot(array[0][hues[0]], pan[0], ms, factor, sampleRate, bitDepth)
	output = output - units[0]
	output2 = triangleDot(array[0][hues[1]], pan[0], ms, factor, sampleRate, bitDepth)
	output2 = output2 - units[1]
	output3 = sawtoothDot(array[0][hues[2]], pan[0], ms, factor, sampleRate, bitDepth)
	output3 = output3 - units[2]
	output = output.overlay(output2)
	output = output.overlay(output3)
	for i in range(1, len(array)):
		factors = array[i]
		units = [int((1-x)*28) for x in factors]
		segment = sineDot(array[i][hues[0]], pan[i], ms, factor, sampleRate, bitDepth)
		segment = segment - units[0]
		segment2 = sawtoothDot(array[i][hues[1]], pan[i], ms, factor, sampleRate, bitDepth)
		segment2 = segment2 - units[1]
		segment3 = triangleDot(array[i][hues[2]], pan[i], ms, factor, sampleRate, bitDepth)
		segment3 = segment3 - units[2]
		segment = segment.overlay(segment2)
		segment = segment.overlay(segment3)
		silence = AudioSegment.silent(50)
		output = output.append(silence, crossfade=0)
		output = output.overlay(segment, position=0.8*len(output))
	return output

def sineDot(value, pan, ms=100, factor=1.0, sampleRate=96000, bitDepth=32):
	cNote = 440
	wave = Sine(cNote*((value+0.00001)*factor)+400, sample_rate=sampleRate, bit_depth=bitDepth)
	output = wave.to_audio_segment(duration=ms)
	output = output.fade_in(int(ms*0.1))
	output = output.fade_out(int(ms*0.1))
	output = output.pan(pan)
	return output

def triangleDot(value, pan, ms=100, factor=1.0, sampleRate=96000, bitDepth=32):
	cNote = 440
	wave = Triangle(cNote*((value+0.0001)*factor), sample_rate=sampleRate, bit_depth=bitDepth)
	output = wave.to_audio_segment(duration=ms)
	output = output.fade_in(int(ms*0.2))
	output = output.fade_out(int(ms*0.2))
	output = output.pan(pan)
	return output

def sawtoothDot(value, pan, ms=100, factor=1.0, sampleRate=96000, bitDepth=32):
	cNote = 440
	wave = Sawtooth(cNote*((value+0.00001)*factor), sample_rate=sampleRate, bit_depth=bitDepth)
	output = wave.to_audio_segment(duration=ms)
	output = output.fade_in(int(ms*0.2))
	output = output.fade_out(int(ms*0.2))
	output = output.pan(pan)
	return output

def getWaveType(index):
	if index == 0:
		return sineDot(1, 0, 800)
	elif index==1:
		return triangleDot(1, 0, 800)
	else:
		return sawtoothDot(1, 0, 800)

def speechAudio(text):
	if not isdir('./audio/temp'):
		makedirs('./audio/temp')
	engine = pyttsx3.init()
	engine.setProperty('rate', 300)
	path = './audio/temp/temp.aiff'
	engine.save_to_file(text, path)
	engine.runAndWait()
	time.sleep(1.0)
	speech = AudioSegment.from_file(path, format='aiff')
	return speech

def exportSound(name, audio, dir):
	if not isdir('./'+dir):
		print('Creating directory ', dir)
		makedirs(dir)
	audio.export('./'+dir+'/'+name+'.mp3', format='mp3')

def compareFrames(arrays):
	n = len(arrays[0])
	sums = np.sum([len(i) for i in arrays])
	return (sums % n) == 0

