import os
import pathlib
import random
from itertools import product
from typing import Union, Tuple, Sequence, List

import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.Image import Resampling
from numpy import ndarray
from scipy import spatial

class ImageHandler:
    """Class to handle some simple image processing"""

    @staticmethod
    def display_images(img_seq: Sequence[ndarray], img_titles: List[str] = []) -> None:
        """Display a sequence of images in a grid"""

        plt.Figure()
        for i, img in enumerate(img_seq):
            plt.subplot(len(img_seq) // 2, 2, i + 1)
            plt.title(img_titles[i])
            plt.imshow(img)
            plt.axis("off")
        plt.show()

    @staticmethod
    def save_images(img_seq: Union[Sequence[ndarray], Sequence[Image.Image]], path: Union[str, pathlib.Path],
                    prefix: str = "", suffix: str = "", extension: str = "png") -> None:
        """Save a sequence of images to a given path"""

        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        if prefix:
            prefix += "_"
        if suffix:
            suffix = f"_{suffix}"

        for i, img in enumerate(img_seq):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img.save(path / f"{prefix}{i}{suffix}.{extension}")
            saved_paths.append(path / f"{prefix}{i}{suffix}.{extension}")
        return saved_paths

class MosaicMaker(ImageHandler):
    """Class to make mosaic (tile) images"""

    def __init__(self, img_width: int = 256, img_height: int = 256, jit_compile: bool = False, seed: int = None):
        self.img_width = img_width
        self.img_height = img_height
        self.jit_compile = jit_compile
        self.seed = seed
        self.model = keras_cv.models.StableDiffusion(img_width=self.img_width, img_height=self.img_height,
                                                     jit_compile=self.jit_compile)
        # Making model's behaviour (more) deterministic
        if self.seed:
            tf.random.set_seed(self.seed)

    def make_images(self, text: str, attributes: Tuple[str] = ("detailed", "painting"),
                    num_images: int = 3) -> Sequence[ndarray]:
        """Make a sequence of images from a given text and attributes using the Stable Diffusion model available from KerasCV submodule"""
        if attributes:
            text = f"{text}, {', '.join(attributes)}"
        return self.model.text_to_image(text, batch_size=num_images)



class MosaicArtMaker(ImageHandler):
    """Class to make a mosaic art from a given image"""

    # Inspired by https://medium.com/@aarongrove/creating-image-mosaics-with-python-8e4c25dd9bf9

    def __init__(self, original_image_path: Union[str, pathlib.Path], tile_images_dir: str = "images/tiles",
                 sqrt_num_tiles: int = 4, tile_file_extension: str = "jpeg"):

        # Load the original image and get its metadata
        self.original_image_path = pathlib.Path(original_image_path)
        self.original_image_name = original_image_path.parts[-1].split(".")[0]
        self.original_image = Image.open(original_image_path)
        self.original_image_width, self.original_image_height = self.original_image.size

        # Check if the original image is a square image
        assert self.original_image_width == self.original_image_height, "Image must be square"

        # Load the tile images and use the mean of the RGB values as the metadata for indexing
        self.sqrt_num_tiles = sqrt_num_tiles
        self.tile_size = self.original_image_width // self.sqrt_num_tiles
        self.grid = list(
            product(range(0, self.original_image_width - (self.original_image_width % self.tile_size), self.tile_size),
                    range(0, self.original_image_height - (self.original_image_height % self.tile_size),
                          self.tile_size)))

        self.tile_images_dir = pathlib.Path(tile_images_dir)
        self.tile_images = []
        for tile_image_path in self.tile_images_dir.glob(f"*.{tile_file_extension}"):
            img = Image.open(tile_image_path)
            img_metadata = []
            for i in range(3):
                img_metadata.append(np.mean(np.array(img)[:, :, i]))
            self.tile_images.append((img, tuple(img_metadata)))

        # Check if the tile images are loaded successfully
        assert self.tile_images, "No tile images found"

        # Creating a KDTree index to find similar tiles to the parts of the original image
        self.index = spatial.KDTree([x[1] for x in self.tile_images])

    def make_mosaic_art(self, k: int = 10, seed: Union[int, None] = None) -> Image.Image:
        """Make a mosaic art from the original image using the tile images created by the SableDiffusion model"""

        # Setting the seed for reproducibility if it is given
        if seed is not None:
            random.seed = seed

        # Making a mosaic art by replacing the parts of the original image with similar tiles from the tile images
        output_image = Image.new('RGB', (self.original_image_width, self.original_image_height))
        for i, j in self.grid:
            box = (j, i, j + self.tile_size, i + self.tile_size)
            original_piece = self.original_image.crop(box)
            original_piece_metadata = []

            # Getting the mean of the RGB values of the original piece to find similar tiles
            for i in range(3):
                original_piece_metadata.append(np.mean(np.array(original_piece)[:, :, i]))

            # Finding the k nearest neighbours of the original piece in the tile images and randomly choosing one
            # So that we can make the mosaic art more interesting by avoiding repetition
            similar_tiles_indices = self.index.query(original_piece_metadata, k=k)[1]

            tile_image = self.tile_images[random.choice(similar_tiles_indices)][0]

            # Resizing the tile image so that it can be pasted on the original image
            tile_image = tile_image.resize((self.tile_size, self.tile_size), Resampling.NEAREST)
            output_image.paste(tile_image, box)
        return output_image


def main(remake_tiles: bool, path: str) -> None:
    """Main function to pack everything together and run it"""

    # Extension to use for saving and loading the tile images
    tile_file_extension = "png"

    # (Re)-make the tile images if the user wants to do so
    if remake_tiles:

        # Create a MosaicMaker object to make the tile images
        image_maker = MosaicMaker(img_width=800, img_height=800, jit_compile=False, seed=33)

        # The text prompts to be used to make the tile images
        prompt_seq = (("A laughing woman", ("realistic", "white background")),
                      ("A sad girl", ("realistic", "white background")),
                      ("An old man", ("realistic", "white background")),
                      ("Face of a sad man", ("realistic", "white background")),
                      ("Drawing of rings of Saturn", ("abstract", "white background")),
                      ("A watercolor painting of a puppy", ("detailed",)),
                      ("Drawing of a red rose", ("elegant", "detailed", "white background")),
                      ("View of a green forest with mountains in the the background", ("elegant", "lush", "nature")),
                      ("A painting of four oranges in a bowl", ("elegant", "detailed", "white background")),
                      ("A ninja shuriken", ("realistic", "metal", "white background")),)

        # Make the tile images and save them
        for index, prompt_data in enumerate(prompt_seq):
            image_seq = image_maker.make_images(prompt_data[0], prompt_data[1], num_images=40)
            image_maker.save_images(img_seq=image_seq, path='images/tiles', prefix=f'p{index}',
                                    extension=tile_file_extension)

    # Create a MosaicArtMaker object with about sqrt_num_tiles*sqrt_num_tiles tiles!
    art_maker = MosaicArtMaker(original_image_path=path, sqrt_num_tiles=100,
                                tile_file_extension=tile_file_extension)

    # Make the mosaic art and save it in the images/outputs directory
    output_image = art_maker.make_mosaic_art(k=10)
    print(f"Created a mosaic art version of '{art_maker.original_image_path}' using "
            f"{art_maker.sqrt_num_tiles * art_maker.sqrt_num_tiles} smaller images created by a Stable Diffusion model")
    paths = art_maker.save_images((output_image,), path='images/outputs',
                            prefix=f'{art_maker.original_image_name}_mosaic_art')

    # Display each original image and its mosaic art version
    art_maker.display_images((art_maker.original_image, output_image),
                                (art_maker.original_image_name, art_maker.original_image_name + "_mosaic_art"))
    return paths[0]

def make_image_square(image_path: pathlib.Path) -> pathlib.Path:
    """画像を正方形に変換して保存"""
    with Image.open(image_path) as img:
        width, height = img.size
        if width != height:
            # 正方形にリサイズ
            new_size = min(width, height)  # 小さい辺に合わせる
            left = (width - new_size) // 2
            top = (height - new_size) // 2
            right = left + new_size
            bottom = top + new_size
            img = img.crop((left, top, right, bottom))
            img.save(image_path)  # 上書き保存
    return
