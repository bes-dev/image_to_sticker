"""
Copyright 2023 by Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from .models import *
from huggingface_hub import hf_hub_download


def image_to_tensor(
        image: NDArray,
        size_tgt: tuple = None,
        normalize_tensor: bool = True
) -> torch.Tensor:
    """ Convert image to tensor

    Args:
        image (np.ndarray): image to convert
        size_tgt (tuple, optional): target size. Defaults to None.
        normalize_tensor (bool, optional): whether to normalize the tensor. Defaults to True.

    Returns:
        torch.Tensor: converted image
    """
    image = torch.from_numpy(image).permute(2, 0, 1)[None, :, :, :] / image.max()
    if size_tgt is not None:
        image = F.upsample(image, size_tgt, mode="bilinear")
    if normalize_tensor:
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


class ImageToStickerPipeline:
    """ Image to sticker pipeline """
    def __init__(
            self,
            model_name: str = "bes-dev/DIS",
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32,
            border_radius: int = 7
    ):
        self.model = ISNetDIS()
        if os.path.exists(os.path.join(model_name, "isnet-general-use.pth")):
            ckpt = torch.load(
                os.path.join(model_name, "isnet-general-use.pth"),
                map_location="cpu"
            )
        else:
            ckpt = torch.load(
                hf_hub_download(model_name, filename="isnet-general-use.pth"),
                map_location="cpu"
            )
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)
        self.device = device
        self.torch_dtype = torch_dtype
        self.borderizer = Borderizer(radius=border_radius)

    def to(self, device: str, dtype: torch.dtype) -> None:
        """ Move the model to the specified device and dtype

        Args:
            device (str): device to move the model to
            dtype (torch.dtype): dtype to move the model to
        """
        self.model.to(device=device, dtype=dtype)
        self.device = device
        self.torch_dtype = dtype

    def postprocess_mask(self, mask: NDArray) -> NDArray:
        """ Postprocess the mask

        Args:
            mask (np.ndarray): mask to postprocess

        Returns:
            np.ndarray: postprocessed mask
        """
        # Find contours on the image
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        # Find the largest contour and extract it
        maxContour = 0
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour = contourSize
                maxContourData = contour
        # Create a mask from the largest contour
        mask_new = np.zeros_like(mask)
        cv2.fillPoly(mask_new, [maxContourData], 1)
        return mask_new

    @torch.inference_mode()
    def compute_mask(self, image: torch.Tensor) -> torch.Tensor:
        """ Compute the mask for the image

        Args:
            image (torch.Tensor): image to compute the mask for

        Returns:
            torch.Tensor: computed mask
        """
        mask = self.model(image)[0][0]
        mask = F.upsample(mask, image.shape[2:], mode="bilinear")
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        return mask

    @torch.inference_mode()
    def __call__(
            self,
            image: NDArray,
            input_size: tuple = None,
            postprocess: bool = False,
            input_format: str = "BGR",
            output_format: str = "BGR"
    ) -> NDArray:
        """ Compute the sticker for the image

        Args:
            image (np.ndarray): image to compute the sticker for
            input_size (tuple, optional): input size for the model. Defaults to None.
            postprocess (bool, optional): whether to postprocess the mask. Defaults to False.
            input_format (str, optional): input format. Defaults to "BGR".
            output_format (str, optional): output format. Defaults to "BGR".

        Returns:
            np.ndarray: computed sticker
        """
        if input_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if input_size is None:
            input_size = image.shape[:2]
        image_t = image_to_tensor(image, input_size, normalize_tensor=True)
        image_t = image_t.to(device=self.device, dtype=self.torch_dtype)
        # compute coarse mask
        mask_coarse = self.compute_mask(image_t)
        image_coarse = mask_coarse * image_t
        # compute fine mask
        mask_fine = self.compute_mask(image_coarse)[0, 0].detach().cpu().float().numpy()
        if postprocess:
            mask_fine = self.postprocess_mask(mask_fine * 255.0).astype(np.uint8) / 255.0
        # compute final image
        image_fine = mask_fine[:, :, None] * image.astype(np.float32)
        image_fine = self.borderizer(image_fine, mask_fine)
        if output_format == "RGBA":
            image_fine = cv2.cvtColor(image_fine, cv2.COLOR_BGRA2RGBA)
        return image_fine


if __name__ == "__main__":
    import argparse
    import os
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model-name", type=str, default="bes-dev/DIS")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=str, default=None)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--border-radius", type=int, default=7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    image_size = tuple(map(int, args.image_size.split(","))) if args.image_size is not None else None
    postprocess = args.postprocess

    image_to_sticker = ImageToStickerPipeline(
        model_name = args.model_name,
        device = device,
        torch_dtype = torch_dtype,
        border_radius = args.border_radius
    )

    for name in tqdm.tqdm(os.listdir(args.input_dir)):
        image = cv2.imread(os.path.join(args.input_dir, name))
        image = image_to_sticker(
            image,
            postprocess = postprocess,
            input_size = image_size
        )
        cv2.imwrite(os.path.join(args.output_dir, name), image)
