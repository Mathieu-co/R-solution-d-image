from PIL import Image

# Chemin vers votre image originale
image_path = 'cousin.jpg'
# Charger l'image avec PIL
img = Image.open(image_path)

# Redimensionner l'image, en réduisant chaque dimension par un facteur
factor = 2
new_width = img.width // factor
new_height = img.height // factor
resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Sauvegarder l'image redimensionnée si nécessaire
resized_img.save('cousin_resized.jpg')

# Maintenant, utilisez 'cousin_resized.jpg' avec votre modèle
