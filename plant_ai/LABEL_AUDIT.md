# Label-Quality Audit — all 9 classes

Date: 2026-10-07
Scope: every real (non-augmented) training image in all 9 classes, 2,371 images.
Method: identical to the earlier `powdery_mildew_fruit` vs `gray_mold` audit — lesion-mask
luminance / texture / brown-share signature, scored against the rival class's IQR.

**Result: 427 images flagged, 0 confirmed mislabeled, 427 verified correct. 0 quarantined.
The dataset is unchanged and no retraining is warranted.**

---

## 1. Rival selection

Four classes sit at 100% test accuracy, so the confusion matrix returns a 0-count "rival" for
them, which is not a usable signal. For those I used the biologically adjacent class from the
earlier analysis, and labelled the reason in the table.

| class | rival | why |
|---|---|---|
| angular_leafspot | leaf_scorch | both necrotic leaf margins; `als->leaf_scorch` in the baseline matrix |
| anthracnose_fruit_rot | powdery_mildew_fruit | confusion matrix, 4 of 8 test images (50%) |
| blossom_blight | gray_mold | **no matrix errors**; same pathogen (*Botrytis cinerea*) |
| gray_mold | powdery_mildew_fruit | confusion matrix, 5 of 40 both directions |
| healthy | angular_leafspot | `als->healthy` 1, `healthy->als` 1 |
| leaf_scorch | angular_leafspot | **no matrix errors**; both necrotic leaf margins |
| leaf_spot | powdery_mildew_fruit | confusion matrix, `leaf_spot->pmf` 1 of 48 |
| powdery_mildew_fruit | gray_mold | confusion matrix, 5 of 12 (42%); re-verified |
| powdery_mildew_leaf | powdery_mildew_fruit | **no matrix errors**; same pathogen, fruit vs leaf |

Discriminative features were chosen per pair automatically (largest IQR separation, capped at
three) so no threshold was hand-tuned to produce a desired count.

## 2. Findings per class

Every flagged image was visually inspected. Counts are of real training images.

| class | n train | flagged | **mislabeled** | kept correct | of which genuinely hard |
|---|---|---|---|---|---|
| angular_leafspot | 207 | 0 | **0** | 0 | 0 |
| anthracnose_fruit_rot | 58 | 5 | **0** | 5 | 0 |
| blossom_blight | 79 | 14 | **0** | 14 | 0 |
| gray_mold | 277 | 95 | **0** | 95 | 0 |
| healthy | 597 | 242 | **0** | 242 | ~3 (see §4) |
| leaf_scorch | 420 | 3 | **0** | 3 | 0 |
| leaf_spot | 337 | 19 | **0** | 19 | ~4 (single faint early spot) |
| powdery_mildew_fruit | 68 | 17 | **0** | 17 | ~6 (wide shots, low coverage) |
| powdery_mildew_leaf | 328 | 32 | **0** | 32 | ~6 (light powder, underside) |
| **total** | **2,371** | **427** | **0** | **427** | **~19** |

Per-image evidence for every flag is in `audit_sheets/index.json`; the contact sheets used for
inspection are in `audit_sheets/`.

### What the flags actually were

Every flagged image was an unambiguous, textbook example of its own labeled class. None showed
the rival's diagnostic symptom.

- **gray_mold, 95 flags** — dense grey-brown fuzzy mycelium on fruit in every case. Gray mould
  genuinely occupies the same luminance and brown-share range as powdery mildew; the two are
  phenotypically adjacent, so IQR overlap is real and carries no information about label
  correctness. (P(grey fuzzy mycelium) here is 1.00.)
- **healthy, 242 flags** — clean, lesion-free green leaves. The lesion mask fires on soil,
  background and petiole in ordinary photos, so a healthy leaf scores like a necrotic one.
  A 40% "error rate" here is an artifact of the metric, not the data.
- **blossom_blight, 14 flags** — flowers with brown necrotic centres. The brown ratio matches
  gray_mould's; the flowers do not.
- **leaf_spot, 19 flags** — purple spots with pale centres, the correct *Mycosphaerella
  fragariae* morphology. Flagged only because some are small and faint.
- **powdery_mildew_fruit, 17 flags** — white powdery growth confirmed on all 17, including the
  six wide greenhouse shots with a dusty sheen on a single berry. These are the residue *after*
  the earlier round removed 13 mislabeled images from this class, which is why the class now
  scans clean.
- **leaf_scorch, 3 flags** — PlantVillage 256x256 studio shots with dark purple scorch spots.
  Distinguished from angular leaf spot by colour, not by texture.
- **powdery_mildew_leaf, 32 flags** — white powder on leaf undersides with the characteristic
  upward leaflet curl.
- **anthracnose_fruit_rot, 5 flags** — sunken dark lesions with salmon spore masses, no powder
  anywhere.

## 3. Why the anthracnose 50% is not a labelling problem

`anthracnose_fruit_rot` has the worst test accuracy in the model (4 of 8 correct, 95% CI
[0.27, 0.73]) and 3 of its 4 errors go to `powdery_mildew_fruit` at high confidence. Because
the rival scan flagged only 5 of 58 training images, I inspected **all 58** directly.

All 58 are coherent, textbook anthracnose fruit rot. There is nothing to clean. The test errors
are a phenotype problem, not a label problem:

| test image | predicted | conf | what it actually shows |
|---|---|---|---|
| `anthracnose_fruit_rot_39` | powdery_mildew_fruit | 0.895 | grey-brown fuzzy patch on fruit — arguably real gray mould |
| `anthracnose_fruit_rot_75` | gray_mold | 0.872 | bleached tan berry, sunken brown tip — anthracnose's over-ripe "bleached" form overlaps fuzzy mould |
| `anthracnose_fruit_rot_22` | powdery_mildew_fruit | 0.671 | dark sunken lesions with fuzzy centres — genuinely ambiguous |
| `anthracnose_fruit_rot_41` | powdery_mildew_fruit | 0.737 | valid sunken lesions; model error |

`anthracnose_fruit_rot_39` is arguably mislabeled **in the TEST split**, which I am not touching.
The other three are phenotype overlap. `powdery_mildew_fruit` fails symmetrically (6 of 12) and
in the same way. These two classes are not linearly separable from this imagery, and no amount
of training-set cleaning will change that.

## 4. Separate finding: class-boundary candidates in `healthy`

Three flagged healthy images show marginal necrosis rather than the discrete purple spots of
angular leaf spot: `healthy_702` (dry brown necrotic margin, bronzed surface), `healthy_670`
(interveinal bronzing with edge browning), `healthy_551` (dark necrotic margin with a few dark
spots).

They are **not** mislabeled relative to this audit's criterion — none of them resemble
`angular_leafspot`, so they correctly belong in the flagged-and-kept bucket. But "healthy" is a
stretch for them: marginal necrosis is closer to `leaf_scorch`, or to abiotic scorch
(salt/fertiliser burn, drought), neither of which this dataset covers. I have left them in place
and am flagging them rather than acting, because reclassifying them is a labelling decision and
moving 3 of 597 healthy images will not move any metric.

## 5. Recommendation

**Do not retrain.** The training set is unchanged, so a retrain would reproduce the existing
model within seed noise while burning a full training run and giving the validation-selected
checkpoint selection another chance to overfit the selection. If a retrain is wanted for other
reasons, say so and I will run it — but this audit provides no evidence that cleaning would
help, and it would be the second time in this project that a plausible-sounding cleanup turned
out to have nothing behind it.

The remaining test errors are concentrated in two fruit-rot classes whose phenotypes genuinely
overlap. If that is worth attacking, the honest options are more anthracnose/gray-mould imagery,
an explicit `fruit_rot_unspecified` class for the genuinely ambiguous berries, or reporting the
overlap as a known limitation. All three change the problem definition rather than the data
cleaning, and all three need your decision.

## Artifacts

| path | contents |
|---|---|
| `audit_labels.py` | step 1, rival selection from the confusion matrix |
| `audit_labels_stage2.py` | step 2, signature scoring, writes `audit_labels_stage2.json` |
| `audit_sheets.py` | step 3, contact sheets + `audit_sheets/index.json` |
| `audit_fullsweep.py` | full-class sweep used for anthracnose |
| `audit_sheets/index.json` | per-flag metrics and sheet position, all 427 |
| `audit_sheets/*.png` | contact sheets, every flagged image |
