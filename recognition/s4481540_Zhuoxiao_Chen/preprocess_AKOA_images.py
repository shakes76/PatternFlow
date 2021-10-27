import lmdb
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as trans_fn
from torchvision import datasets
from io import BytesIO
from functools import partial
import multiprocessing
import argparse
# import the required libraries for the data pre-processing.


def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, size, Image.LANCZOS)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)

    return i, out


def pre_process_AKOA_images(AKOA_image, raw_set, number_cpus, dimensions=(8, 16, 32, 64, 128, 256, 512, 1024)):
    """
    This function is used to simply resize the images into multiple dimensions,
    and store the resized images into a dict structure, 
    whethe dimension is the key, image is the value to be easily called and obtained. 
    """
    method = partial(resize_worker, sizes=dimensions)

    AKOA_images = sorted(raw_set.imgs, key=lambda x: x[0])
    AKOA_images = [(index, image) for index, (image, label) in enumerate(AKOA_images)]
    count = 0

    with multiprocessing.Pool(number_cpus) as pool:
        for index, AKOA_images in tqdm(pool.imap_unordered(method, AKOA_images)):
            for dim, AKOA_image in zip(dimensions, AKOA_images):
                key = f'{dim}-{str(index).zfill(5)}'.encode('utf-8')
                AKOA_image.put(key, AKOA_image)
            count += 1
        AKOA_image.put('length'.encode('utf-8'), str(count).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-process the AKOA dataset for further training the Progressive StyleGAN.')
    parser.add_argument('AKOA_raw_images_path', type=str)
    parser.add_argument('--number_worker', type=int, default=8)
    parser.add_argument('--output_directory', type=str)
    args = parser.parse_args()

    raw_AKOA_images = datasets.ImageFolder(args.AKOA_raw_images_path)
    with lmdb.open(args.output_directory, map_size=1024 ** 4, readahead=False) as environment:
        with environment.begin(write=True) as data:
            pre_process_AKOA_images(data, raw_AKOA_images, args.number_worker)
