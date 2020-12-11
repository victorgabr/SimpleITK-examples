# encoding: utf-8
"""An example to read in a image file, and resample the image to a new grid.
"""

import SimpleITK as sitk
import os
from glob import glob
from tqdm import tqdm
import numpy as np


# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_sitk_image(
    sitk_image: sitk.Image, new_size, interpolator="gaussian", fill_value=0
) -> sitk.Image:
    """
        modified version from:
            https://github.com/jonasteuwen/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """

    # if pass a path to image
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)

    assert (
        interpolator in _SITK_INTERPOLATOR_DICT.keys()
    ), "`interpolator` should be one of {}".format(_SITK_INTERPOLATOR_DICT.keys())

    if not interpolator:
        interpolator = "linear"
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                "Set `interpolator` manually, "
                "can only infer for 8-bit unsigned or 16, 32-bit signed integers"
            )

    #  8-bit unsigned int
    if sitk_image.GetPixelIDValue() == 1:
        # if binary mask interpolate it as nearest
        interpolator = "nearest"

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]
    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()

    # new spacing based on the desired output shape
    new_spacing = tuple(
        np.array(sitk_image.GetSpacing())
        * np.array(sitk_image.GetSize())
        / np.array(new_size)
    )

    # setup image resampler - SimpleITK 2.0
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetDefaultPixelValue(orig_pixelid)
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetDefaultPixelValue(fill_value)
    # run it
    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image


def resample_image_to_voxel_size(
    image: sitk.Image, new_spacing, interpolator, fill_value=0
) -> sitk.Image:
    """
        resample a 3d image to a given voxel size (dx, dy, dz)
    :param image: 3D sitk image
    :param new_spacing: voxel size (dx, dy, dz)
    :param interpolator:
    :param fill_value: pixel value when a transformed pixel is outside of the image.
    :return:
    """

    # computing image shape based on the desired voxel size
    orig_size = np.array(image.GetSize())
    orig_spacing = np.array(image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = orig_size * (orig_spacing / new_spacing)

    #  Image dimensions are in integers
    new_size = np.ceil(new_size).astype(np.int)

    #  SimpleITK expects lists, not ndarrays
    new_size = [int(s) for s in new_size]

    return resample_sitk_image(image, new_size, interpolator, fill_value)



def main(REGEX_TO_IMAGES, WRITE_TO):
    for image in tqdm(glob(REGEX_TO_IMAGES)):
        tqdm.write('Resampling {}'.format(image))
        resampled_image = resample_image_to_voxel_size(
            image, spacing=[1, 1, 1],
            interpolator='linear', 
            fill_value=-1200
        )
        base_name = 'resampled_' + os.path.basename(image)
        write_to = os.path.join(WRITE_TO, base_name)
        tqdm.write('Writing resampled image to {}'.format(write_to))
        sitk.WriteImage(resampled_image, write_to, True)  # True = compress image.


if __name__ == '__main__':
    base_path = './datasets/'
    write_path = './resampled/'

    subpaths = ['path1/mhd/*.mhd', 'path2/*.nii.gz']
    write_paths = ['path1/', 'path2/']

    for i, sub in enumerate(subpaths):
        main(base_path + sub, write_path + write_paths[i])

