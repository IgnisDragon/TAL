# TAL

Implement of temporal activity localization (TAL) on tensorflow

Include :
- Temporal Activity Localization via Language (TALL)
- C3D (pretrain on sport1M)
- I3D (testing, pretrain on kinetics-600)
- skip-thoughts

## CTRL

Cross-modal Temporal Regression Localizer (CTRL) include : 

- Visual Encoder :
C3D extract feature for video clips

- Sentence Encoder :
skip-thoughts encoder extract embeddings

- Multi-modal processing network :
generate combined representations for visual and text domain

- Temporal regression network :
produce alignment scores and location offsets between the input sentence query and video clips

## ROLE

cRoss-modal mOment Localization nEtwork (ROLE), composing a language-temporal attention network, a multi-modal processing, and a MLP module.

- Language-Temporal Attention Network : 
use Bi-directional LSTM to encode the whole query. each word in query project into embedding vector via GloVe.
attention Network feed by word representation and temporal moment contexts

-  Multi-modal processing : 
generate cross-modal representation by concate temporal visual moment and textual embedding

- MLP network : 
strong representation power of non-linear hidden layers enables complicated interactions among the features of the cross-modal representation

## ARCN

Attentive Cross-Modal Retrieval Network (ACRN) comprises of the following components:
- Memory Attention Network : 
leverages the weighting contexts to enhance the visual embedding of each moment

- Cross-modal fusion network : 
explores the intramodal and the inter-modal feature interactions to generate the moment-query representations

- Regression network : 
estimates the relevance scores and predicts the location offsets of the golden moments

# Reference
TALL: Temporal Activity Localization via Language Query

https://openaccess.thecvf.com/content_ICCV_2017/papers/Gao_TALL_Temporal_Activity_ICCV_2017_paper.pdf

Cross-modal Moment Localization in Videos

https://www.researchgate.net/profile/Meng-Liu-67/publication/328374995_Cross-modal_Moment_Localization_in_Videos/links/6052a318a6fdccbfeae93f75/Cross-modal-Moment-Localization-in-Videos.pdf

Attentive Moment Retrieval in Videos

http://staff.ustc.edu.cn/~hexn/papers/sigir18-video-retrieval.pdf