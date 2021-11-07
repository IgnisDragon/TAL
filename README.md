# TAL

Implement of temporal activity localization (TAL) on tensorflow

Include :
- Temporal Activity Localization via Language (TALL)
- C3D
- skip-thoughts


Cross-modal Temporal Regression Localizer (CTRL) include : 

- Visual Encoder :
C3D extract feature for video clips(pretrain on sport1M)
- Sentence Encoder :
skip-thoughts encoder extract embeddings
- Multi-modal processing network :
generate combined representations for visual and text domain
- Temporal regression network :
produce alignment scores and location offsets between the input sentence query and
video clips
