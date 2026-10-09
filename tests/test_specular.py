"""specular_removal.py (Tan & Ikeuchi 2005 반사광 분리) 검증."""
import cv2
import numpy as np
import pytest

from specular_removal import SpecularSeparator, make_synthetic_scene


@pytest.fixture(scope="module")
def scene():
    img, truth, spec = make_synthetic_scene(width=640, height=480)
    diffuse, specular, sf = SpecularSeparator().separate(img)
    return img, truth, spec, diffuse, specular, sf


def mean_err(a, truth, mask):
    return np.abs(a.astype(np.int16) - truth).mean(axis=2)[mask].mean()


def test_outputs_shape_and_dtype(scene):
    img, _, _, diffuse, specular, sf = scene
    for out in (diffuse, specular, sf):
        assert out.shape == img.shape and out.dtype == np.uint8


def test_removes_highlights_on_colored_paint(scene):
    img, truth, spec, diffuse, _, _ = scene
    paint = (spec > 0.05) & (cv2.cvtColor(truth, cv2.COLOR_BGR2HSV)[..., 1] > 80)
    before, after = mean_err(img, truth, paint), mean_err(diffuse, truth, paint)
    assert after < before / 5, (before, after)


def test_leaves_highlight_free_areas_untouched(scene):
    img, truth, spec, diffuse, _, _ = scene
    calm = spec < 0.01
    assert np.abs(diffuse.astype(np.int16) - img)[calm].mean() < 1.0


def test_input_equals_diffuse_plus_specular(scene):
    img, _, _, diffuse, specular, _ = scene
    recon = diffuse.astype(np.int16) + specular
    assert np.abs(recon - img).max() <= 1  # specular = 입력 - 난반사 (0 미만은 0)


def test_white_light_on_single_color_is_recovered():
    """균일한 유색 표면 + 가운데에 백색광을 더한 패치: 가운데가 주변 난반사 색으로 돌아와야 한다."""
    diffuse = np.zeros((120, 120, 3), np.uint8)
    diffuse[:] = (40, 90, 200)                       # 주황빛 도료 (BGR)
    yy, xx = np.mgrid[0:120, 0:120]
    white = 50 * np.exp(-((yy - 60) ** 2 + (xx - 60) ** 2) / 300.0)
    img = np.clip(diffuse + white[..., None], 0, 255).astype(np.uint8)
    out, spec, _ = SpecularSeparator(analysis_scale=1.0).separate(img)
    centre = (slice(50, 70), slice(50, 70))
    assert np.abs(out[centre].astype(int) - diffuse[centre]).mean() < 4
    assert spec[centre].mean() > 30


def test_achromatic_surface_is_left_alone():
    """회색 강판은 난반사와 광원 색이 같아 분리 불가 -> 손대지 않아야 한다 (모델의 한계)."""
    gray = np.full((100, 100, 3), 120, np.uint8)
    gray[40:60, 40:60] = 200
    out, spec, _ = SpecularSeparator().separate(gray)
    assert np.array_equal(out, gray)
    assert spec.max() == 0


@pytest.mark.parametrize("value", [0, 255])
def test_black_and_white_out_images_are_stable(value):
    img = np.full((80, 80, 3), value, np.uint8)
    outs = SpecularSeparator().separate(img)
    for out in outs:
        assert out.shape == img.shape
        assert np.isfinite(out.astype(np.float32)).all()


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        SpecularSeparator(target_chroma=0.3)
    with pytest.raises(ValueError):
        SpecularSeparator(illum_chroma=(1, 0, 1))
    with pytest.raises(ValueError):
        SpecularSeparator().separate(np.zeros((10, 10), np.uint8))
