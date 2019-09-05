import PIL.Image
import sys

args = sys.argv[1:]

img = PIL.Image.open(args[0])
img = img.convert('RGB')
img = img.resize([512, 512])
img.save(args[1])
