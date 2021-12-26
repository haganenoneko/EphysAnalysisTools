# EphysAnalysisTools
Tools for analysis of electrophysiology data. 

# Outline 
The goals of this library are as follows:

## Leak subtraction
- data:
  - automated detection of a consecutive pair of symmetric linear ramps (default)
  - user-specified (via interactive matplotlib interface) region of the recording
- equation:
  - linear Ohmic $I_{leak} = \gamma_{leak}(V_m - E_{leak})$
  - cubic polynomial
  - GHK with sodium, potassium, or chloride terms
  - user-specified function
- fitting method:
  - least squares Levenberg-Marquedt method (`lmfit`)
  - user-specified argument, corresponding to algorithm in `lmfit`
- diagnostics:
  - Standard plot of local diagnostics (2 x N)
    - 1st row (sweep # x dependent variable)
      - N parameter estimates, 
      - $r^2$ (Pearson for linear fit, Spearman else) for
        - raw current x voltage 
        - fit x raw current 
    - 2nd row 
      - original, leak fit, and leak-subtracted current of region used for fitting
  - Global (recording-wide) diagnostics
    
    Imperfect leak subtraction leads to, e.g. inward currents becoming outward, and indicates nonlinear leak. Current amplitudes within $\Delta t$ ms (default: 500 ms) of a voltage step should be $\leq 2\%$ of the maximal current drop. 

## Reversal potential and I-V curves
- Data:
  - instantaneous peak current amplitudes of the first varying-voltage step of a protocol
  - user-specified region (interactive via Matplotlib):
    - ramp 
    - varying-voltage step
- Equation: linear, cubic polynomial, or GHK (see [Leak subtraction](#leak-subtraction))
- fitting method: see [Leak subtraction](#leak-subtraction)
- diagnostics:
  - plot I-V for leak subtracted current and current density
## Activation curves
- Data:
  - 
