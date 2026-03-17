# Neural Optimizer Lab

Interactive Streamlit app to visualize and compare optimization algorithms on 2D loss landscapes.

## What this project shows

- How different optimizers move on non-convex and anisotropic surfaces
- Behavior differences between a Foundation set and a Modern set of optimizers
- Scenario-based benchmarking with noise and different learning rates
- Convergence information shown in the final sandbox legend

## Files

- `neural_optimz.py`: main Streamlit app with Sandbox and Benchmark tabs
- `code1.py`: earlier/simple optimizer comparison prototype

## Optimizers included

Foundation set:
- SGD
- SGD + Momentum
- AdaGrad
- RMSProp
- Adam

Modern set:
- Adam
- AdamW
- Lion
- Sophia

## Loss landscapes

- Bumpy Egg-Crate (non-convex, many local minima)
- Elliptical Canyon (ill-conditioned quadratic)

## Requirements

- Python 3.9+
- `streamlit`
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install streamlit numpy matplotlib
```

## Run

From the `vis_opti` folder:

```bash
streamlit run neural_optimz.py
```

## Recommended VS Code workflow

1. Select your Python interpreter (for example, `streamlit_app` conda env).
2. Open terminal and activate env.
3. Run Streamlit command above.
4. Use Sandbox tab for parameter tuning and final-frame screenshot capture.

## Screenshot notes for article writing

- Use Sandbox tab for clean visual narratives.
- Final frame legend includes convergence iteration tags.
- Hyperparameter controls are compact for easier full-panel screenshots.

## Troubleshooting

- If breakpoints are not hit, launch using VS Code debugger (F5) with a Streamlit launch config.
- If you see environment mismatch, re-select interpreter and open a new terminal.
- If app fails after edits, run:

```bash
python -m py_compile neural_optimz.py
```

## License

No license file is currently included in this folder. Add one before public distribution if needed.
