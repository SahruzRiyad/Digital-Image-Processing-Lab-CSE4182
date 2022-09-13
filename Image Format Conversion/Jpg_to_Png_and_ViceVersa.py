from PIL import Image


img = Image.open("scenary2.jpeg")
img.save("scenary2.png")


img = Image.open("Ludo_img.png")
rgb_img = img.convert('RGB') 
rgb_img.save("Ludo_img.jpeg")
