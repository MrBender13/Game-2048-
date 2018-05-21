from PIL import Image

for letter in 'wasd':
    img = Image.open(letter + '.png')
    for x in range(img.size[1]):
        for y in range(img.size[0]):
            r, g, b = img.getpixel((y, x))
            if r > 200 and g > 200 and b > 200:
                img.putpixel((y, x), (240, 240, 240))
            else:
                img.putpixel((y, x), (0, 0, 0))

    img.save(letter + '.png')

"""    img = Image.open(letter + '_red.png')
    for x in range(img.size[1]):
        for y in range(img.size[0]):
            r, g, b = img.getpixel((y, x))
            if r > 200 and g > 200 and b > 200:
                img.putpixel((y, x), (240, 240, 240))
            else:
                img.putpixel((y, x), (0, 0, 0))

    img.save(letter + '_red.png')"""
