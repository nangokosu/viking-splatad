"""Export a trained SplatAD run's Gaussians as a SuperSplat-readable 3DGS ``.ply`` (T82-T86).

SplatAD stores a learned 16-d feature per Gaussian, not RGB; colour comes from the upstream
``RGBDecoderCNN``. The modules here bake that decoder per Gaussian, fit spherical harmonics, crop
SplatAD's init background points and write the PLY (``scripts/22_export_supersplat_ply.py``).
"""
