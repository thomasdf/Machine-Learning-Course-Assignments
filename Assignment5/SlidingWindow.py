from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from Assignment5 import Preprocess, Classifier

base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"

def slicy(img, size: int):
	return slide(img, size, size)


def shady(img: Image, size: int, coor, predictions, treshold: float = .70, scaled_shade: bool = True):
	c = zip(coor, predictions)
	return shade2d(img, c, size, 255, treshold, scaled_shade)


def slide(array: np.ndarray, stride: int, size: int):
	coordinates = []
	slices = []

	for y in range(0, len(array) - size, stride):

		for x in range(0, len(array[y]) - size, stride):
			arr = array[y:(y + size), x:(x + size)]

			# a = draw2d(a, x, y, (x + size), (y + size))
			a = Preprocess.preprocessArr(arr)
			coordinates.append((x, y))
			slices.append(a)
	stacked_slice = np.vstack(slices)
	return coordinates, stacked_slice

def shade2d(im: Image, classified_img, size: int, intensity: int = 1, treshold: float = .90,
            scaled_shader: bool = True):
	im = im.convert("RGB")
	chars = "abcdefghijklmnopqrstuvwxyz"
	object = [
		(0, 0, 255, intensity), #A = shade of Red
		(50, 0, 0, intensity), #B
		(75, 0, 0, intensity), #C
		(100, 0, 0, intensity),
		(125, 0, 0, intensity),
		(150, 0, 0, intensity),
		(175, 0, 0, intensity),
		(200, 0, 0, intensity),
		(225, 0, 0, intensity),
		(250, 0, 0, intensity),
		(255, 0, 0, intensity),
		(0, 25, 0, intensity), # L = Shade of Green
		(0, 50, 0, intensity),
		(0, 75, 0, intensity),
		(0, 100, 0, intensity),
		(0, 125, 0, intensity),
		(0, 150, 0, intensity),
		(0, 175, 0, intensity),
		(0, 200, 0, intensity),
		(0, 225, 0, intensity),
		(0, 250, 0, intensity),
		(0, 25, 0, intensity), #V = shade of blue
		(0, 50, 0, intensity),
		(0, 75, 0, intensity),
		(0, 100, 0, intensity),
		(0, 125, 0, intensity),
	]
	dont_use_treshhold = treshold == -1
	rect = Image.new('RGBA', (size, size))
	# charim = Image.new('RGBA', (size, size))
	# fnt = ImageFont.truetype(assignment5dir + "/res/" + 'font.ttf', 25)
	#fnt = ImageFont.load_default()
	pdraw = ImageDraw.Draw(rect)
	# chardraw = ImageDraw.Draw(charim)
	for xy, cl in classified_img:
		x, y = xy
		offset = (x, y)
		object_index = cl.argmax()
		scale = cl[object_index]
		if (dont_use_treshhold or scale >= treshold):
			# object_index = cl.argmax()

			color = list(object[object_index])
			color[3] = int(255 * scale) if scaled_shader else color[3]
			pdraw.rectangle([0, 0, size, size], fill=tuple(color), outline=object[object_index])
			# chardraw.text((1, 1), chars[object_index], None, fnt)
			im.paste(rect, offset, mask=rect)
			# im.paste(charim, offset, mask=charim)
	return im


def shade2dgrayscale(im: Image, classified_img, size: int, intensity: int = 1, treshold: float = .90,
            scaled_shader: bool = True):
	im = im.convert("RGB")
	object = [
		(0, 0, 0, intensity)
	]
	dont_use_treshhold = treshold == -1
	rect = Image.new('RGBA', (size, size))
	# charim = Image.new('RGBA', (size, size))
	# fnt = ImageFont.truetype(assignment5dir + "/res/" + 'font.ttf', 25)
	# fnt = ImageFont.load_default()
	pdraw = ImageDraw.Draw(rect)
	# chardraw = ImageDraw.Draw(charim)
	for xy, cl in classified_img:
		x, y = xy
		offset = (x, y)
		object_index = cl.argmax()
		scale = cl[object_index]
		if (dont_use_treshhold or scale >= treshold):
			# object_index = cl.argmax()

			color = list(object[0])
			color[3] = int(255 * scale) if scaled_shader else color[3]
			pdraw.rectangle([0, 0, size, size], fill=tuple(color), outline=object[0])
			# chardraw.text((1, 1), chars[object_index], None, fnt)
			im.paste(rect, offset, mask=rect)
		# im.paste(charim, offset, mask=charim)
	return im


def shade2dWchars(im: Image, classified_img, size: int, intensity: int = 1, treshold: float = .90,
            scaled_shader: bool = True):
	im = im.convert("RGB")
	chars = "abcdefghijklmnopqrstuvwxyz"
	object = [
		(0, 0, 255, intensity),  # A = shade of Red
		(50, 0, 0, intensity),  # B
		(75, 0, 0, intensity),  # C
		(100, 0, 0, intensity),
		(125, 0, 0, intensity),
		(150, 0, 0, intensity),
		(175, 0, 0, intensity),
		(200, 0, 0, intensity),
		(225, 0, 0, intensity),
		(250, 0, 0, intensity),
		(255, 0, 0, intensity),
		(0, 25, 0, intensity),  # L = Shade of Green
		(0, 50, 0, intensity),
		(0, 75, 0, intensity),
		(0, 100, 0, intensity),
		(0, 125, 0, intensity),
		(0, 150, 0, intensity),
		(0, 175, 0, intensity),
		(0, 200, 0, intensity),
		(0, 225, 0, intensity),
		(0, 250, 0, intensity),
		(0, 25, 0, intensity),  # V = shade of blue
		(0, 50, 0, intensity),
		(0, 75, 0, intensity),
		(0, 100, 0, intensity),
		(0, 125, 0, intensity),
	]
	dont_use_treshhold = treshold == -1
	#rect = Image.new('RGBA', (size, size))
	#fnt = ImageFont.load_default()
	#pdraw = ImageDraw.Draw(rect)
	for xy, cl in classified_img:
		x, y = xy
		offset = (x, y)
		object_index = cl.argmax()
		scale = cl[object_index]
		if (dont_use_treshhold or scale >= treshold):
			fnt = ImageFont.truetype(assignment5dir + "/res/" + 'font.ttf', 12)
			drawim = Image.new('RGBA', (size, size))
			draw = ImageDraw.Draw(drawim)
			# object_index = cl.argmax()

			color = list(object[0])
			color[3] = int(255 * scale) if scaled_shader else color[3]
			draw.rectangle([0, 0, size, size], fill=tuple(color), outline=object[0])
			draw.text((2, 1), chars[object_index], fill="#FFFFFF", outline="000000", font=fnt)
			# im.paste(rect, offset, mask=rect)
			im.paste(drawim, offset, mask=drawim)
	return im


def sliding_classify(img, arr, size: int, stride=20, model=Classifier, treshold: float = .70,
                     scaled_shade: bool = True, shader: callable = shade2d, trained = None):
	coor, slices = slide(arr, stride, size)
	r = model.run(slices, trained)
	c = zip(coor, r)
	a = shade(img, c, size, 100, treshold, scaled_shade, shader=shader)

	return a


def shade(img, classified_imgarr, size, stride, threshold, scaled_shade, shader=shade2d):
	return shader(img, classified_imgarr, size, stride, threshold, scaled_shade)

