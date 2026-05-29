import numpy as np

from segmentation.patches import SlidingWindowSampler, _resolve_window_stride


def _count_for(h, w, **kw):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return SlidingWindowSampler(**kw).sample(img).num_patches


def test_no_bounds_is_unchanged():
    # without bounds the configured window/stride are used verbatim
    n = _count_for(600, 600, window_size=96, stride=48)
    assert n == _count_for(600, 600, window_size=96, stride=48)
    # 600 with window 96 stride 48 -> 12 positions per dim -> 144
    assert n == 144


def test_max_patches_caps_large_image():
    # a big image would explode; cap keeps it under the ceiling
    n = _count_for(4000, 3000, window_size=96, stride=48, max_patches=500)
    assert n <= 500
    assert n > 0


def test_min_patches_densifies_small_image():
    # a modest image under-samples at the configured stride; min pulls it up
    n = _count_for(500, 500, window_size=96, stride=200, min_patches=100)
    assert n >= 100


def test_bounds_are_respected_across_sizes():
    for (h, w) in [(300, 300), (800, 1200), (5000, 4000)]:
        n = _count_for(h, w, window_size=96, stride=48, min_patches=80, max_patches=400)
        assert 80 <= n <= 400, (h, w, n)


def test_resolver_returns_configured_when_unbounded():
    ws, s = _resolve_window_stride(600, 600, 96, 48, None, None)
    assert (ws, s) == (96, 48)


def test_resolver_raises_on_inverted_bounds():
    import pytest

    with pytest.raises(ValueError):
        _resolve_window_stride(600, 600, 96, 48, min_patches=500, max_patches=100)


def test_tiny_image_shrinks_window_for_min():
    # image barely larger than window: even stride 1 can't reach min -> window shrinks.
    # 100x100 with a 96 window gives only 5x5=25 positions at stride 1, < 50.
    ws, s = _resolve_window_stride(100, 100, 96, 48, min_patches=50, max_patches=None)
    assert ws < 96  # had to shrink the window
    assert s == 1
