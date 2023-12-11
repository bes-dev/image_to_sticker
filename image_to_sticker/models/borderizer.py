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
import cv2
import numpy as np
from numpy.typing import NDArray


class Borderizer:
    """ Add border to image """
    def __init__(self, radius: int = 16):
        self.radius = radius
        self.kernel = np.fromfunction(
            lambda x, y: ((x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2) * 1,
            (2 * radius + 1, 2 * radius + 1),
            dtype=int,
        ).astype(np.uint8)

    def __call__(self, image: NDArray, mask: NDArray) -> NDArray:
        """ Add border to image

        Args:
            image: image [B, G, R]
            mask: mask [0, 1]

        Returns:
            image with border [B, G, R, A]
        """
        image, mask = image / image.max(), mask / mask.max()
        mask_ext = cv2.dilate(mask.copy(), self.kernel, iterations=1)
        border = mask_ext - mask
        image = image + cv2.merge([border, border, border], 3)
        image = (image * 255).astype(np.uint8)
        mask = (mask_ext * 255).astype(np.uint8)
        image = cv2.merge([*cv2.split(image), mask], 4)
        return image
