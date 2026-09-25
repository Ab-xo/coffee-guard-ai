from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from coffeeguard.inference.predictor import softmax
from coffeeguard.inference.preprocess import preprocess, resize_for_model
from coffeeguard.inference.quality import quality_metrics
from tests.fixtures.synthetic import make_leaf


def test_blur_lowers_sharpness_and_darkening_lowers_brightness():
    img = make_leaf("Cercospora", seed=1, size=(512, 384))
    q = quality_metrics(img)
    blurred = quality_metrics(img.filter(ImageFilter.GaussianBlur(4)))
    dark = quality_metrics(Image.eval(img, lambda v: v // 4))
    assert blurred["sharpness"] < q["sharpness"] / 5
    assert dark["brightness"] < q["brightness"] / 2
    assert 0.1 < q["green_frac"] < 0.9


def test_blank_image_has_no_green_and_no_detail():
    q = quality_metrics(Image.new("RGB", (300, 300), (128, 128, 128)))
    assert q["green_frac"] == 0.0
    assert q["sharpness"] == 0.0


def test_preprocess_shape_dtype_and_exif_rotation():
    img = make_leaf("Healthy", seed=2, size=(640, 480))
    x = preprocess(img, 224)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    # an EXIF 'rotate 90°' tag (orientation 6) must be applied: 640x480 becomes portrait
    import io

    from coffeeguard.inference.quality import to_rgb

    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    with Image.open(io.BytesIO(buf.getvalue())) as tagged:
        assert to_rgb(tagged).size == (480, 640)
        assert resize_for_model(tagged, 224).size == (224, 224)


def test_softmax_temperature_flattens():
    logits = np.array([[4.0, 1.0, 0.0, -1.0]])
    p1 = softmax(logits)
    p2 = softmax(logits, temperature=2.0)
    assert np.isclose(p1.sum(), 1.0) and np.isclose(p2.sum(), 1.0)
    assert p2.max() < p1.max()
