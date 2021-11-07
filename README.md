# TAL

Implement of temporal activity localization (TAL)

Include :
- Temporal Activity Localization via Language (TALL)
- C3D
- skip-thoughts


Cross-modal Temporal Regression Localizer (CTRL) include : 

- Visual Encoder :
use C3D to extract feature (C3D pretrain on sport1M)
- Sentence Encoder :
use skip-thoughts encoder sentence as feature
- Multi-modal processing network :
generate combined representations for visual and text domain
- Temporal regression network :
produce alignment scores and location offsets between the input sentence query and
video clips
