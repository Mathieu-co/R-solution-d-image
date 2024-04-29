import argparse
import os
import time
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, ToPILImage
from model import Generator
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy


def combine_images(bicubic_image, hr_image, sr_image):
    # Ajouter un cadre noir à chaque image
    border_size = 10  # Définir la taille du cadre
    bicubic_image = ImageOps.expand(bicubic_image, border=border_size, fill='black')
    hr_image = ImageOps.expand(hr_image, border=border_size, fill='black')
    sr_image = ImageOps.expand(sr_image, border=border_size, fill='black')

    # Créer une nouvelle image pour tenir les trois images côte à côte
    total_width = bicubic_image.width + hr_image.width + sr_image.width - 2 * border_size
    new_image = Image.new('RGB', (total_width, hr_image.height))

    # Coller chaque image dans la nouvelle image
    new_image.paste(bicubic_image, (0, 0))
    new_image.paste(hr_image, (bicubic_image.width - border_size, 0))  # ajuster pour l'overlap
    new_image.paste(sr_image, (bicubic_image.width + hr_image.width - 2 * border_size, 0))  # ajuster pour l'overlap

    return new_image


def process_and_save_images(image_folder, model_path, upscale_factor, output_folder='output_images'):
    model = Generator(upscale_factor).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(model_path))

    os.makedirs(output_folder, exist_ok=True)

    moyenne_ssim = 0
    moyenne_psnr = 0
    i = 0

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            hr_image = Image.open(image_path)
            lr_image = hr_image.resize((hr_image.width // upscale_factor, hr_image.height // upscale_factor),
                                       Image.BICUBIC)

            # Générer l'image bicubique à partir de l'image LR redimensionnée à la taille HR
            bicubic_image = lr_image.resize(hr_image.size, Image.BICUBIC)

            # Préparer l'image LR pour le modèle
            image_tensor = ToTensor()(lr_image).unsqueeze(0)
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            # Appliquer le modèle
            with torch.no_grad():
                start_time = time.perf_counter()
                sr_tensor = model(image_tensor)
                elapsed_time = time.perf_counter() - start_time
                print(f"Processing time for {image_name}: {elapsed_time:.3f}s")

            # Convertir le tenseur de sortie en image PIL et la redimensionner à la taille de HR
            sr_image = ToPILImage()(sr_tensor[0].cpu())
            sr_image = sr_image.resize(hr_image.size, Image.BICUBIC)

            # Combiner les images bicubique, HR, et SR
            combined_image = combine_images(bicubic_image, hr_image, sr_image)

            # Convertir les images PIL en tableau numpy pour le calcul des métriques
            hr_array = numpy.array(hr_image)
            sr_array = numpy.array(sr_image)

            # Assurer que la taille de la fenêtre est adéquate pour les images
            win_size = min(3, hr_image.width, hr_image.height)
            if win_size % 2 == 0:  # s'assurer que win_size est impair
                win_size -= 1

            # Calculer SSIM et PSNR
            ssim_value = ssim(hr_array, sr_array, data_range=sr_array.max() - sr_array.min(), multichannel=True,
                              win_size=win_size)
            psnr_value = psnr(hr_array, sr_array, data_range=sr_array.max() - sr_array.min())

            moyenne_ssim += ssim_value
            moyenne_psnr += psnr_value

            i += 1

            print(f"SSIM: {ssim_value:.4f}")
            print(f"PSNR: {psnr_value:.4f}")

            # Sauvegarder l'image combinée
            combined_image_path = os.path.join(output_folder, f"image_{upscale_factor}_{image_name}")
            combined_image.save(combined_image_path)
            print(f"Combined image saved to {combined_image_path}")
    moyenne_ssim /= i
    moyenne_psnr /= i

    print(f"SSIM moyen pour {i} images de test est : {moyenne_ssim:.4f}")
    print(f"PSNR moyen pour {i} images de test est : {moyenne_psnr:.4f}")


model_path = 'epochs/netG_epoch_2_50.pth'
output_folder = 'output_images'
image_folder = 'Image_test'
upscale_factor = 2

process_and_save_images(image_folder, model_path, upscale_factor, output_folder)
