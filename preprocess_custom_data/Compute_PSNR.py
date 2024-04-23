from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 读取影像
image = img_as_float(io.imread('./image/neus-barn-encoding.png'))

height = image.shape[0] // 2
image2 = image[:height]
image1 = image[height:]

# 计算PSNR
psnr_value = psnr(image1, image2)

# 计算SSIM
ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min(),channel_axis=2)

print("PSNR: ", psnr_value)
print("SSIM: ", ssim_value)