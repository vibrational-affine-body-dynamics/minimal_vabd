# Weight Generation

Run from the repository root in the appropriate conda environment:

```powershell
# Fast-CD scalar weights (MATLAB is used when available; SciPy is the fallback)
python -m weight_gen.fast_cd --weights 20 --backend auto

# Linear modes plus modal-warping rotational features
python -m weight_gen.modal_warping --backend auto

# PCA of translation and rotation features into the final scalar weight basis
python -m weight_gen.basis_span --kind translation-rotation --weights 20
```

Add `--viewer` to any stage that offers it when an interactive Polyscope window
is useful. `lma.m` and `nullspace.m` contain the original MATLAB generalized
eigenanalysis. `--backend scipy` runs without MATLAB; `--backend matlab` makes a
MATLAB error fatal rather than falling back.
