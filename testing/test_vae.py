import argparse
import os
from extensions_built_in.dataset_tools.tools.image_tools import load_image
import torch
from torchvision.transforms import Resize, ToTensor
from diffusers import AutoencoderKL
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)
    return images


def paramiter_count(model):
    state_dict = model.state_dict()
    paramiter_count = 0
    for key in state_dict:
        paramiter_count += torch.numel(state_dict[key])
    return int(paramiter_count)


def calculate_metrics(vae, images, max_imgs=-1, save_output=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    rfid_scores = []
    psnr_scores = []
    lpips_scores = []

    # transform = transforms.Compose([
    #     transforms.Resize(256, antialias=True),
    #     transforms.CenterCrop(256)
    # ])
    # needs values between -1 and 1
    to_tensor = ToTensor()
    
    # remove _reconstructed.png files
    images = [img for img in images if not img.endswith("_reconstructed.png")]

    if max_imgs > 0 and len(images) > max_imgs:
        images = images[:max_imgs]

    for img_path in tqdm(images):
        try:
            img = load_image(img_path, force_rgb=True)
            # img_tensor = to_tensor(transform(img)).unsqueeze(0).to(device)
            img_tensor = to_tensor(img).unsqueeze(0).to(device)
            img_tensor = 2 * img_tensor - 1
            # if width or height is not divisible by 8, crop it
            if img_tensor.shape[2] % 8 != 0 or img_tensor.shape[3] % 8 != 0:
                img_tensor = img_tensor[:, :, :img_tensor.shape[2] // 8 * 8, :img_tensor.shape[3] // 8 * 8]

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue


        with torch.no_grad():
            reconstructed = vae.decode(vae.encode(img_tensor).latent_dist.sample()).sample

        # Calculate rFID
        # rfid = fid_score.calculate_frechet_distance(vae, img_tensor, reconstructed)
        # rfid_scores.append(rfid)

        # Calculate PSNR
        psnr_val = psnr(img_tensor.cpu().numpy(), reconstructed.cpu().numpy())
        psnr_scores.append(psnr_val)

        # Calculate LPIPS
        lpips_val = lpips_model(img_tensor, reconstructed).item()
        lpips_scores.append(lpips_val)

    # avg_rfid = sum(rfid_scores) / len(rfid_scores)
    avg_rfid = 0
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    
    if save_output:
        filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        folder = os.path.dirname(img_path)
        save_path = os.path.join(folder, filename_no_ext + "_reconstructed.png")
        reconstructed = (reconstructed + 1) / 2
        reconstructed = reconstructed.clamp(0, 1)
        reconstructed = transforms.ToPILImage()(reconstructed[0].cpu())
        reconstructed.save(save_path)

    return avg_rfid, avg_psnr, avg_lpips


def main():
    parser = argparse.ArgumentParser(description="Calculate average rFID, PSNR, and LPIPS for VAE reconstructions")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the VAE model")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--max_imgs", type=int, default=-1, help="Max num of images. Default is -1 for all images.")
    # boolean store true
    parser.add_argument("--save_output", action="store_true", help="Save the output images")
    args = parser.parse_args()

    if  os.path.isfile(args.vae_path):
        vae = AutoencoderKL.from_single_file(args.vae_path)
    else:
        try:
            vae = AutoencoderKL.from_pretrained(args.vae_path)
        except:
            vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae.eval()
    vae = vae.to(device)
    print(f"Model has {paramiter_count(vae)} parameters")
    images = load_images(args.image_folder)

    avg_rfid, avg_psnr, avg_lpips = calculate_metrics(vae, images, args.max_imgs, args.save_output)

    # print(f"Average rFID: {avg_rfid}")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average LPIPS: {avg_lpips}")


if __name__ == "__main__":
    main()
