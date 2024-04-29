import cv2
import os

def resize_images(images_folder, output_folder, scale_factor):
    """
    Redimensionne toutes les images dans le dossier 'images_folder' et les enregistre dans 'output_folder'
    avec un facteur d'échelle donné.

    Args:
    - images_folder (str): Chemin vers le dossier contenant les images haute résolution.
    - output_folder (str): Chemin vers le dossier où enregistrer les images redimensionnées.
    - scale_factor (float): Facteur d'échelle pour la réduction de la résolution.
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Liste des fichiers dans le dossier des images
    image_files = os.listdir(images_folder)

    # Redimensionner chaque image
    for image_file in image_files:
        # Charger l'image
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        # Obtenir les nouvelles dimensions
        new_width = int(image.shape[1] / scale_factor)
        new_height = int(image.shape[0] / scale_factor)

        # Redimensionner l'image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Modifier le nom de fichier pour inclure le facteur d'échelle
        filename, ext = os.path.splitext(image_file)
        output_filename = f"{filename}_factor{scale_factor}{ext}"

        # Enregistrer l'image redimensionnée
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, resized_image)


# Chemins des dossiers d'entrée et de sortie
HR_images_folder = "Image_test"
LR_images_folder = "Resultat_test_image"

# Facteur d'échelle
scale_factor = 2

# Redimensionner les images
resize_images(HR_images_folder, LR_images_folder, scale_factor)