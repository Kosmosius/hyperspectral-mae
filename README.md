# hyperspectral-mae

spectralae/
  core/                    # physics, numerics, data models (NO torch here)
    __init__.py
    types.py               # dataclasses/TypedDicts for SRF, Grid, Spectrum, Sample
    grid.py                # CanonicalGrid, masks, ranges
    srf.py                 # SRF curves, param surrogates, projection basis
    renderer.py            # NumPy/Scipy: R matrix, quadrature, cache
    basis.py               # B-spline banks, orthonormalization, versioning
    cr.py                  # continuum removal (eval), smoothed variant (train)
    hashing.py             # SHA-256 helpers for assets
  io/
    __init__.py
    manifests.py           # JSON schema, validation
    datasets/
      lab.py               # USGS/ECOSTRESS/ASTER
      field.py
      satellite.py         # PRISMA/EnMAP pixel samplers
    loaders.py             # unified Dataset API, collate fns
  usr/
    __init__.py
    tokenizer.py           # tokens = [y | RFF(c,Δ) | s]
    rff.py                 # banks seeded & persisted
  ml/
    __init__.py
    interfaces.py          # Encoder, Decoder, Model Protocols/ABCs
    modules/
      encoders/            # torch only
        adapter_transformer.py
        set_transformer.py
        relpos_bias.py
      decoders/
        bspline.py
        deeponet.py
        residual_line.py
    losses/
      band_likelihood.py   # heteroscedastic Huber/Student-t
      spectral_losses.py   # SAM, curvature, box, CR loss (torch wrappers)
      invariance.py        # GRL head, HSIC/MMD
    model.py               # SpectralAE(nn.Module): wires encoder/decoder/render wrapper
    render_torch.py        # thin wrapper calling core.renderer, returns torch tensors
  train/
    runner.py              # train/eval loop orchestration (DDP-aware)
    schedules.py           # masking/emulation/GRL ramps
    emulation.py           # random + catalog-biased emulators
    ema.py                 # teacher/student
    seed.py                # determinism
  eval/
    metrics.py             # SAM, per-λ RMSE, curvature, bins, ECE
    reports.py             # tables/plots, CI artifacts
  cli/
    fit.py                 # train entrypoint
    eval.py                # evaluation suite
    render.py              # physics sanity tools
    export.py              # checkpoint/export QA
  configs/                 # Hydra or Pydantic YAMLs grouped by domain
    defaults.yaml
    grid/
    srf/
    model/
    train/
    data/
  assets/
    basis/                 # versioned B-matrices + meta (hashes, knots, grid)
    rff/
    srf_bases/
  tests/
    unit/                  # pure fast tests
    golden/                # parity against saved CSVs (renderer/CR)
    integration/           # toy training, invariance smoke
  scripts/
    build_bases.py         # generate persisted bases with hashes
    build_srf_cache.py     # project SRFs to basis, cache + hash
    sanity_renderer.py
