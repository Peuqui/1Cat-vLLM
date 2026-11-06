# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.media import ImageMediaIO

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


def test_image_media_io_rgba_custom_background(tmp_path):
    """Test RGBA to RGB conversion with custom background colors."""
    # Create a simple RGBA image with transparent and opaque pixels
    rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))  # Red with full opacity

    # Make top-left quadrant transparent
    for i in range(5):
        for j in range(5):
            rgba_image.putpixel((i, j), (0, 0, 0, 0))  # Fully transparent

    # Save the test image to tmp_path
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Test 1: Default white background (backward compatibility)
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    default_numpy = np.array(converted_default)

    # Check transparent pixels are white
    assert default_numpy[0][0][0] == 255  # R
    assert default_numpy[0][0][1] == 255  # G
    assert default_numpy[0][0][2] == 255  # B
    # Check opaque pixels remain red
    assert default_numpy[5][5][0] == 255  # R
    assert default_numpy[5][5][1] == 0  # G
    assert default_numpy[5][5][2] == 0  # B

    # Test 2: Custom black background via kwargs
    image_io_black = ImageMediaIO(rgba_background_color=(0, 0, 0))
    converted_black = image_io_black.load_file(test_image_path)
    black_numpy = np.array(converted_black)

    # Check transparent pixels are black
    assert black_numpy[0][0][0] == 0  # R
    assert black_numpy[0][0][1] == 0  # G
    assert black_numpy[0][0][2] == 0  # B
    # Check opaque pixels remain red
    assert black_numpy[5][5][0] == 255  # R
    assert black_numpy[5][5][1] == 0  # G
    assert black_numpy[5][5][2] == 0  # B

    # Test 3: Custom blue background via kwargs (as list)
    image_io_blue = ImageMediaIO(rgba_background_color=[0, 0, 255])
    converted_blue = image_io_blue.load_file(test_image_path)
    blue_numpy = np.array(converted_blue)

    # Check transparent pixels are blue
    assert blue_numpy[0][0][0] == 0  # R
    assert blue_numpy[0][0][1] == 0  # G
    assert blue_numpy[0][0][2] == 255  # B

    # Test 4: Test with load_bytes method
    with open(test_image_path, "rb") as f:
        image_data = f.read()

    image_io_green = ImageMediaIO(rgba_background_color=(0, 255, 0))
    converted_green = image_io_green.load_bytes(image_data)
    green_numpy = np.array(converted_green)

    # Check transparent pixels are green
    assert green_numpy[0][0][0] == 0  # R
    assert green_numpy[0][0][1] == 255  # G
    assert green_numpy[0][0][2] == 0  # B


def test_image_media_io_rgba_background_color_validation():
    """Test that invalid rgba_background_color values are properly rejected."""

    # Test invalid types
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color="255,255,255")

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=255)

    # Test wrong number of elements
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255, 255, 255))

    # Test non-integer values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255.0, 255.0, 255.0))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, "255", 255))

    # Test out of range values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(256, 255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, -1, 255))

    # Test that valid values work
    ImageMediaIO(rgba_background_color=(0, 0, 0))  # Should not raise
    ImageMediaIO(rgba_background_color=[255, 255, 255])  # Should not raise
    ImageMediaIO(rgba_background_color=(128, 128, 128))  # Should not raise


def test_image_media_io_load_bytes(tmp_path):
    """Test load_bytes with valid and invalid image data."""
    # Save a valid RGB image to use as source bytes
    valid_image = Image.new("RGB", (8, 8), (100, 150, 200))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    valid_data = valid_path.read_bytes()

    # Test 1: Valid image bytes load successfully and are fully decoded
    image_io = ImageMediaIO()
    result = image_io.load_bytes(valid_data)

    # Check the returned media is a properly loaded image
    assert isinstance(result.media, Image.Image)
    assert result.media.size == (8, 8)
    assert result.media.getpixel((0, 0)) == (100, 150, 200)

    # Test 2: Garbage bytes raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"not an image")

    # Test 3: Truncated PNG header raises ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    # Test 4: Real PNG truncated mid-stream raises ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(valid_data[: len(valid_data) // 2])

    # Test 5: Empty bytes raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"")


def test_image_media_io_load_file(tmp_path):
    """Test load_file with valid and invalid image files."""
    # Save a valid RGB image to disk
    valid_image = Image.new("RGB", (4, 4), (10, 20, 30))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    # Test 1: Valid image file loads successfully and is fully decoded
    image_io = ImageMediaIO()
    result = image_io.load_file(valid_path)

    # Check the returned media is a properly loaded image
    assert isinstance(result.media, Image.Image)
    assert result.media.size == (4, 4)
    assert result.media.getpixel((0, 0)) == (10, 20, 30)

    # Test 2: File with garbage content raises ValueError
    bad_file = tmp_path / "bad.png"
    bad_file.write_bytes(b"this is not an image")

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(bad_file)

    # Test 3: File with truncated PNG header raises ValueError
    truncated_file = tmp_path / "truncated.png"
    truncated_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(truncated_file)

    # Test 4: Real PNG file truncated mid-stream raises ValueError
    valid_data = valid_path.read_bytes()
    truncated_real_file = tmp_path / "truncated_real.png"
    truncated_real_file.write_bytes(valid_data[: len(valid_data) // 2])

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(truncated_real_file)


def _left_right_image(image_format: str, exif: bytes | None = None) -> bytes:
    """16x8 image, left half red and right half blue."""
    image = Image.new("RGB", (16, 8), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 8, 8))
    buffer = BytesIO()
    if exif is None:
        image.save(buffer, format=image_format, quality=95)
    else:
        image.save(buffer, format=image_format, quality=95, exif=exif)
    return buffer.getvalue()


def test_image_media_io_applies_exif_orientation():
    """A JPEG with EXIF orientation 3 (180 degrees) is loaded upright."""
    exif = Image.Exif()
    exif[0x0112] = 3  # Orientation
    result = ImageMediaIO().load_bytes(_left_right_image("JPEG", exif.tobytes()))

    red, _, blue = result.media.getpixel((2, 4))
    assert blue > 200 and red < 60  # the blue half is now on the left


def test_image_media_io_keeps_image_with_broken_exif():
    """Broken EXIF data does not fail the request; the image stays as stored."""
    broken_exif = b"Exif\x00\x00" + b"\x00" * 8  # no valid TIFF header
    data = _left_right_image("PNG", broken_exif)
    with pytest.raises(SyntaxError):
        Image.open(BytesIO(data)).getexif()

    result = ImageMediaIO().load_bytes(data)

    assert result.media.getpixel((2, 4)) == (255, 0, 0)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, (255, 255, 255)), ({"rgba_background_color": [0, 255, 0]}, (0, 255, 0))],
)
def test_image_media_io_palette_trns_uses_background(kwargs, expected):
    """Transparent palette entries of a P-mode PNG get the background color."""
    image = Image.new("P", (4, 4), 1)
    image.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
    image.putpixel((0, 0), 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG", transparency=1)

    result = ImageMediaIO(**kwargs).load_bytes(buffer.getvalue())

    assert result.media.mode == "RGB"
    assert result.media.getpixel((0, 0)) == (255, 0, 0)
    assert result.media.getpixel((1, 1)) == expected
