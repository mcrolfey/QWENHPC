# Asbestos detection with a fine-tuned VLM: microscope vs smartphone images

Oct 31, 2026 · @mick

A fine-tuned Qwen2.5-VL-7B model detects asbestos with 0.909 balanced accuracy on microscope images and 0.952 on macro smartphone images. The accuracy gap (0.907 vs 0.952) is not statistically significant (Fisher's exact test, p = 0.094). Neither model works on the other image type. Retrained with synthetic copies of its training images, the macro model scored 1.000 on the same test images, which makes the gap significant (p = 3.0 × 10⁻⁶). It also scored 1.000 on the 99 test images that have no near-duplicate in the training set, and again 1.000 after the macro images were re-split so that near-duplicates never cross splits (the original macro model scored 0.960 there). The microscope model's highest-scoring tiles contain annotated fibres far more often than chance (within-image AUC 0.825), but tiles without annotated fibres in asbestos images also score high, and inspection showed that these often hold fibres the annotators did not box. Naming the asbestos type, the macro model with synthetic data is correct on 148 of 151 test images (balanced accuracy 0.919). The best microscope type model reaches balanced accuracy 0.590 over seven classes once its decisions are corrected for the class mix it saw in training; in the classes both datasets share it trails the macro model (balanced accuracy 0.592 vs 0.943; accuracy 0.732 vs 0.980, p = 2.5 × 10⁻¹⁵).

## Results at a glance

Both models detect asbestos well on their own image type; the microscope model needed a tuned rule for combining its tile predictions to get there. Rules and thresholds were chosen on the validation split, then applied once to the test split.

| Image type | Test images | Decision rule (chosen on validation) | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Macro (smartphone) | 145 | P(asbestos) ≥ 0.50 | 0.952 | 0.952 | 0.947 | 0.957 | 0.986 |
| Microscopic | 1,781 | Mean of the 3 highest tile scores ≥ 0.953 | 0.907 | 0.909 | 0.891 | 0.928 | 0.971 |

Accuracy difference, microscopic minus macro: −0.044 (Fisher's exact test, p = 0.094). With 145 macro test images the macro accuracy's 95% interval is 0.917–0.986, so a gap of this size cannot be ruled in or out at this sample size.

## Data

11,818 microscope images and 995 macro smartphone images were usable, both close to balanced between asbestos and non-asbestos.

| Dataset | Source | Annotations | Images found | Images used | Asbestos | Non-asbestos | Test images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Microscopic | `combinedImgs/images` | YOLO boxes, 7 classes | 12,076 | 11,818 | 6,639 | 5,179 | 1,781 |
| Macro (smartphone) | `combinedImgs/macro_asbestos` | LabelMe boxes, 4 classes | 1,000 | 995 | 498 | 497 | 145 |

Excluded: 144 microscope images with no label file, 114 with an empty label file, and 5 macro images with no annotation.

**Image labels.** Each image's class was derived from its bounding boxes: the asbestos class with the largest total box area if any asbestos box was present, otherwise the largest non-asbestos class. 1,342 images contained boxes of more than one class. For the macro set, the asbestos / non-asbestos folder was used as a cross-check; no image contradicted its folder. All `A-*` classes count as asbestos and all `NA-*` classes as non-asbestos.

| Dataset | Class | Code | Images | Binary label |
| --- | --- | --- | --- | --- |
| Microscopic | Chrysotile Pointy | A-CP | 3,884 | Asbestos |
| Microscopic | Chrysotile Fluffy | A-CF | 2,498 | Asbestos |
| Microscopic | Crocidolite | A-CRO | 117 | Asbestos |
| Microscopic | Chrysotile and Organic Fiber | A-COF | 114 | Asbestos |
| Microscopic | Amosite | A-AM | 26 | Asbestos |
| Microscopic | Organic Fiber | NA-OF | 3,752 | Non-asbestos |
| Microscopic | Cement Sheet | NA-CS | 1,427 | Non-asbestos |
| Macro | Chrysotile | A-CRY | 414 | Asbestos |
| Macro | Chrysotile, Organic Fiber and Cement Sheet | A-CRY-OF-CS | 46 | Asbestos |
| Macro | Chrysotile and Organic Fiber | A-CRY-OF | 38 | Asbestos |
| Macro | Cement Sheet | NA-CS | 497 | Non-asbestos |

**Splits.** Each dataset was split 70 / 15 / 15 into train, validation and test, stratified by class. Splits were made per image: no specimen identifier was available, so images of the same physical sample can fall into different splits (see Limitations).

## Training

Two separate models were fine-tuned from Qwen2.5-VL-7B-Instruct with QLoRA: one on microscope images, one on macro images. Both used the same recipe; only the image handling differed. Both were trained on the detection (Yes/No) question only; the later macro models (synthetic data, rebuilt splits) were trained on both the detection and the type question, and a separate microscope model was trained on the type question alone (1 epoch, 4 tiles per image).

**Task framing.** Detection was posed as a one-word question-answering task. Each example is a chat: a system prompt ("You are an expert asbestos analyst…"), then the image and the question, then the answer the model learns to give.

- Microscope tile: "Does this region cropped from a microscope image of a material sample show asbestos? Answer with exactly one word: 'Yes' or 'No'."
- Macro image: the same question about "this smartphone photograph of a building material".
- The loss is computed on the answer tokens only; the prompt and image tokens are masked out.

**Model and adapters (QLoRA).**

| Setting | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-VL-7B-Instruct |
| Language model weights | 4-bit NF4, double quantization, bf16 compute (bitsandbytes) |
| Vision encoder | kept in bf16 and frozen (not quantized, not adapted) |
| LoRA targets | q, k, v, o, gate, up and down projections of every language-model layer |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Trained parameters | LoRA adapters only; all base weights frozen |

**Optimisation (both models).**

| Setting | Value |
| --- | --- |
| Optimiser | paged AdamW, 8-bit |
| Learning rate | 1e-4, cosine decay, 3% warm-up |
| Batch | 2 per step × 8 gradient-accumulation steps = 16 |
| Precision / memory | bf16, gradient checkpointing, max gradient norm 1.0 |
| Class balance | inverse-frequency sampling of Yes and No examples |
| Augmentation | random horizontal and vertical flips each time an image is loaded; also 90° rotations for microscope tiles |
| Model selection | validation loss after each epoch; best checkpoint kept; early stopping if validation loss did not improve for 3 evaluations |

The offline, bounding-box-preserving augmentation built into the pipeline was not used in these runs: its job did not complete before training, so both models trained on the original images plus the on-the-fly flips above.

**Image handling.**

|  | Microscopic model | Macro model |
| --- | --- | --- |
| Input unit | 512 × 512 px tiles, 10% overlap (about 61 tiles per image) | whole image |
| Visual-token budget | 512 × 512 px per tile (\~330 tokens) | up to 1,024 tokens (≈ 0.8 MP) per image |
| Tiles used in training | up to 4 per image, tiles with annotated fibres first | – |
| Training label | from the boxes inside each tile (a box counts if ≥ 30% of it is in the tile); tiles of asbestos images that contain no annotated fibre were left out as ambiguous | the image label |
| Epochs | 1 | up to 3 |
| Wall time (one GPU) | 10 h 08 min, of which about 4 h was the validation pass over \~108,000 tiles | 28.5 min |
| Final validation loss | 0.351 | – |

Tiling keeps fibres at full resolution; down-scaling a whole micrograph into the model's token budget would make thin fibres disappear. Labelling tiles from their own boxes avoids teaching the model that a fibre-free tile of an asbestos sample "shows asbestos".

## Evaluation method

Each test image received a score between 0 and 1 for "asbestos present", and a decision rule chosen on the validation split turned scores into Yes/No.

1. **Scoring.** The model reads the same prompt as in training. Its next-token probabilities for "Yes" and "No" are renormalised over these two answers, giving P(asbestos). Every image gets a valid score; no free-text answers have to be parsed.
2. **Microscope tiles.** Every tile of every test image is scored, not just 4: about 109,000 tiles per run (5 h 54 min on one GPU). The tile scores of an image are then combined into one image score.
3. **Choosing the rule and threshold.** Twelve combining rules were compared (any tile, mean, quantiles, mean of the top-k tiles, fraction of tiles above a cut-off). For each rule, the threshold with the best balanced accuracy on the validation split was found. The rule with the best validation balanced accuracy was kept and applied once to the test split. The macro model went through the same procedure; with one score per image, only its threshold was tuned, and it stayed at 0.50.
4. **Metrics.** Accuracy, balanced accuracy (mean of sensitivity and specificity), sensitivity, specificity, ROC-AUC and Cohen's κ. 95% confidence intervals come from 1,000 bootstrap resamples of the test images.
5. **Comparisons.** Two models on the same test images: McNemar's test on paired correct/incorrect outcomes. Microscope vs macro: the test sets contain different specimens, so the comparison is unpaired, using Fisher's exact test on accuracy.

The test split was not used for any choice: rules, thresholds and the training checkpoint were all selected on validation data.

## Combining microscope tile predictions

The choice of rule for combining \~61 tile scores per image moved the microscope result from 0.614 to 0.909 balanced accuracy, without retraining.

**Why the default rule failed.** The default called an image asbestos if any one tile had P(asbestos) ≥ 0.50. With about 61 chances per image, almost every non-asbestos image had at least one false-alarm tile:

| Rule | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Any tile ≥ 0.50 (default) | 0.658 | 0.614 | 1.000 | 0.227 | 0.962 |
| Mean of top 3 tiles ≥ 0.953 (chosen on validation) | 0.907 | 0.909 | 0.891 | 0.928 | 0.971 |

The ROC-AUC of 0.962 under the default rule shows that the image scores already separated the classes well. Only the decision point was misplaced: for images with many tiles, "asbestos" has to mean that the most confident tiles are very confident, not that one tile passes 0.50.

**All rules compared.** Sorted by validation balanced accuracy, the selection criterion. Test columns were not used for the choice.

| Rule | Threshold | Validation AUC | Validation balanced acc. | Test AUC | Test balanced acc. | Test sensitivity | Test specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mean of top 3 tiles (selected) | 0.953 | 0.962 | 0.899 | 0.971 | 0.909 | 0.891 | 0.928 |
| Mean of top 5 tiles | 0.924 | 0.961 | 0.895 | 0.970 | 0.904 | 0.892 | 0.915 |
| Any tile (max) | 0.974 | 0.952 | 0.892 | 0.962 | 0.891 | 0.902 | 0.879 |
| 95th percentile of tiles | 0.915 | 0.961 | 0.891 | 0.970 | 0.906 | 0.880 | 0.931 |
| Fraction of tiles ≥ 0.9 | 0.057 | 0.952 | 0.890 | 0.967 | 0.905 | 0.913 | 0.896 |
| Mean of top 10 tiles | 0.872 | 0.953 | 0.878 | 0.962 | 0.892 | 0.858 | 0.926 |
| 90th percentile of tiles | 0.819 | 0.955 | 0.876 | 0.963 | 0.898 | 0.885 | 0.911 |
| Fraction of tiles ≥ 0.7 | 0.171 | 0.946 | 0.867 | 0.956 | 0.879 | 0.881 | 0.877 |
| Fraction of tiles ≥ 0.5 | 0.222 | 0.928 | 0.848 | 0.936 | 0.846 | 0.915 | 0.777 |
| Mean of all tiles | 0.258 | 0.926 | 0.842 | 0.935 | 0.851 | 0.877 | 0.825 |
| 75th percentile of tiles | 0.485 | 0.929 | 0.842 | 0.938 | 0.858 | 0.854 | 0.863 |
| Median tile | 0.085 | 0.862 | 0.764 | 0.874 | 0.768 | 0.828 | 0.708 |

- Rules that listen to the few most confident tiles (top-3 or top-5 mean, high percentiles, any tile with a high threshold) all reach 0.89–0.91 test balanced accuracy; the differences between them are small.
- Rules that average over all tiles do worse (0.85 for the mean, 0.77 for the median). This fits asbestos fibres appearing in only a few tiles of a positive image.
- Validation and test rankings agree closely, so the selected rule is not a lucky pick on validation.

## Where the microscope model looks

Image accuracy does not show whether a tile is called asbestos because of the fibres in it. The saved tile scores of the microscope test run were checked against the bounding boxes, without retraining. A tile holding at least 30% of an asbestos box counts as a fibre tile (the rule used to label training tiles); a tile that no asbestos box touches counts as empty. The 1,772 tiles that overlap a box by less than 30% are left out.

| Tiles | Count | Mean P(asbestos) |
| --- | --- | --- |
| Fibre tiles (asbestos images) | 2,829 | 0.852 |
| Empty tiles in asbestos images | 55,018 | 0.432 |
| Tiles of non-asbestos images | 49,532 | 0.154 |

| Measure | Result (95% CI) | By chance |
| --- | --- | --- |
| Fibre tiles vs tiles of non-asbestos images (AUC) | 0.962 (0.957–0.966) | 0.5 |
| Fibre vs empty tiles in the same images (AUC, pooled) | 0.825 (0.815–0.836); mean per image 0.828 | 0.5 |
| Top tile overlaps an asbestos box (993 images) | 0.432 (0.403–0.464) | 0.080 |
| Share of the top 3 tiles overlapping a box | 0.358 | 0.080 |

- **The model does localise.** Within an image, a fibre tile outranks an empty tile 83% of the time, and the top tile lands on an annotated fibre 5.4 times as often as a randomly chosen tile (p < 0.0001, 10,000 random draws per image). The within-image comparison ignores anything that differs between images, such as stain, magnification or background.
- **High-scoring empty tiles hold fibres that were not boxed.** Empty tiles in asbestos images average 0.432, nearly three times the 0.154 of tiles in non-asbestos images, and in 57% of asbestos images the top tile touches no box. Overlay images of 10 of these misses were inspected: their top tiles showed visible fibres without a box. The boxes mark some of the fibres in each image, not all, so the model is finding real fibres that the annotation does not count.
- **The image decision rests on fibres, boxed or not.** The selected rule averages the three highest tiles. Only 36% of those overlap a box, but the inspected misses show that the rest often hold unboxed fibres.

Because the boxes are not exhaustive, some tiles counted as empty hold fibres, so the within-image AUC (0.825) and the pointing game (0.432) are lower bounds on how well the model localises. Training was not misled by the missing boxes: tiles of asbestos images without a box were left out, not labelled negative. Measuring localisation exactly would need a sample of test images with every fibre boxed.

## Cross-domain results

Neither model detects asbestos on the other image type: both score at chance (Cohen's κ ≤ 0.03). These runs used the default rule (any tile ≥ 0.50 for microscope images, P(asbestos) ≥ 0.50 for macro images).

| Model | Tested on | Test images | Accuracy (95% CI) | Balanced accuracy | Cohen's κ |
| --- | --- | --- | --- | --- | --- |
| Macro | Macro images | 145 | 0.952 (0.917–0.986) | 0.952 | 0.903 |
| Microscopic | Macro images | 145 | 0.490 (0.414–0.573) | 0.504 | 0.009 |
| Microscopic | Microscope images | 1,781 | 0.658 (0.635–0.679) | 0.614 | 0.247 |
| Macro | Microscope images | 1,781 | 0.558 (0.534–0.581) | 0.512 | 0.026 |

- On macro images, the macro model beats the microscope model by 0.462 accuracy (McNemar, p = 2.5 × 10⁻¹⁴).
- On microscope images, the microscope model beats the macro model by 0.100 under the default rule (McNemar, p = 3.2 × 10⁻²³). The microscope model's row is its default-rule result; with the rule chosen on validation it reaches 0.907.
- The macro model's scores on microscope images were not re-combined with a tuned rule, so its 0.558 may understate what it could reach; its ROC-AUC on these images has not been checked.

A model trained on one image type therefore cannot stand in for the other, most likely because magnification and appearance differ so much between the two.

## Macro model with synthetic training data

Retraining the macro model on synthetic copies of its training images raised it from 0.952 to 1.000 accuracy on the same 145 test images. It stays at 1.000 on the test images that have no near-duplicate in training, and on splits rebuilt so that near-duplicates stay together (where the original model scores 0.960). That makes the microscope–macro gap significant, although the comparison now mixes image type with a larger difference in training data.

**How the synthetic data was made.** Each macro training image received at least 9 synthetic copies, and the rarer type classes (A-CRY-OF, A-CRY-OF-CS) up to 20, giving about ten times as many training images. The copies come from the pipeline's offline augmentation, which transforms each image and its bounding boxes together, so the boxes stay on the material:

- Geometry: horizontal flips, 90° rotations, scaling ±15%, rotation ±20°, shifts ±5%, mild perspective change.
- Capture conditions: brightness, contrast, colour, saturation and gamma jitter, motion or Gaussian blur, JPEG compression, noise.
- A box that ends up less than 40% visible is dropped, and a copy that loses all of its boxes is discarded.
- Only the training split was augmented. Validation and test images are the originals, so the test set is the same 145 images as before.
- Box overlays of 30 synthetic copies were checked by eye before training.

The model was trained with the same settings as the original macro model, plus the type question (QLoRA, up to 3 epochs, best checkpoint on validation loss; 10 h 08 min on one GPU), and the decision threshold was again chosen on the validation split.

| Macro model | Decision rule (chosen on validation) | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Original | P(asbestos) ≥ 0.50 | 0.952 | 0.952 | 0.947 | 0.957 | 0.986 |
| With synthetic data | P(asbestos) ≥ 0.958 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- **Against the original macro model** (same 145 test images, paired): +0.048 accuracy. The augmented model got right all 7 images the original got wrong and none the other way round (McNemar exact test, p = 0.016). The two models also differ in training task: the original was trained on detection only, the synthetic-data model on detection and type, so this gain mixes the two changes. The rebuilt-split comparison below, where both models were trained on both questions, isolates the synthetic data (+0.041, p = 0.0625).
- **Against the microscope model** (unpaired): 0.907 vs 1.000, a difference of −0.093 (Fisher's exact test, p = 3.0 × 10⁻⁶). With the original macro model this gap was not significant (p = 0.094).
- **On microscope images** the augmented macro model is still at chance: accuracy 0.543, balanced accuracy 0.503, Cohen's κ = 0.007 (any tile ≥ 0.50, 1,781 images). Synthetic phone-photo variation does not carry over to micrographs.
- The validation split was also perfectly separated, so any threshold in a wide range would have given the same result; 0.958 is simply the one the search returned.

**Near-duplicate check.** Splits were made per image, so photos of the same piece of material could sit in both training and test, letting a model recognise the specimen rather than the asbestos. Every validation and test image was compared with every training image using a 64-bit perceptual hash, under all flips and 90° rotations, to find its closest training image. Synthetic copies were left out of this comparison. A Hamming distance of 4 or less (out of 64) was treated as a near-duplicate.

- 46 of 145 test images (32%) and 33 of 152 validation images (22%) have a near-duplicate in training; 17 test images are at distance 2 or less. Every one of these pairs shares its label.
- Loosening the cut-off to 10 matches 82% of test images, but at that distance pairs are not necessarily the same specimen.

| Test images | Images | Original macro model | With synthetic data |
| --- | --- | --- | --- |
| With a near-duplicate in training | 46 | 0.935 (3 errors) | 1.000 |
| Without a near-duplicate | 99 | 0.960 (4 errors) | 1.000 |
| All | 145 | 0.952 (7 errors) | 1.000 |

The overlap does not explain the results:

- The original model was not more accurate on near-duplicate images (0.935 vs 0.960), so there is no sign that recognising specimens drives the scores.
- The synthetic-data model is perfect on the 99 images without a near-duplicate. Its gain came from fixing errors in both groups (4 among them, 3 among near-duplicates). With 99 images, 99 out of 99 correct puts accuracy at about 0.963 or higher (95%).

One caveat remains. A perceptual hash only catches near-identical photos, not different photos of the same specimen, so specimens could still be shared across splits; only a specimen id recorded during data collection would rule that out. The other concern, that near-identical photos make the test set easier, is removed by rebuilding the splits (below).

**Splits rebuilt so near-duplicates stay together.** The macro images were re-split: images within distance 4 of each other, directly or through a chain of matches, formed one group, and whole groups were assigned to training, validation or test, stratified by class. 329 of the 995 macro images fell into 104 groups (the largest has 13 images). 450 images changed split, and no near-duplicate pair now crosses splits. The new split has 691 training, 153 validation and 151 test images, with every class in each. Both macro models were retrained on it, with the synthetic copies regenerated from the new training split, and their thresholds were chosen on the new validation split. The microscope split and model are unchanged.

| Model | Test images | Accuracy (95% CI) | Errors |
| --- | --- | --- | --- |
| Original macro model | 151 | 0.960 (0.929–0.986) | 6 (3 false positives, 3 false negatives) |
| Macro model with synthetic data | 151 | 1.000 | 0 |
| Microscope model (unchanged) | 1,781 | 0.907 | 165 |

- **Synthetic vs original macro model** (same test images, paired by near-duplicate group, 121 groups): +0.041 accuracy (95% CI 0.008–0.083). All 5 discordant groups favour the synthetic-data model, but McNemar's exact test gives p = 0.0625, just above 0.05. Counted per image (6 errors vs 0) the exact test would give p = 0.031, but images in one group are not independent, so the group-level test is the one to report.
- **Microscope vs original macro model**: 0.907 vs 0.960, Fisher's exact test p = 0.025.
- **Microscope vs synthetic-data macro model**: 0.907 vs 1.000, p = 1.1 × 10⁻⁶.

Keeping near-duplicates together did not lower either macro model's score: the original model went from 0.952 to 0.960 and the synthetic-data model stayed at 1.000. The test set is different, so these are not paired with the earlier numbers, but they confirm that near-identical photos across splits were not inflating the results. On the rebuilt split the microscope–macro gap is significant even for the original macro model (p = 0.025, against p = 0.094 before), mostly because its macro score is slightly higher. The synthetic-data model's gain over the original is consistent across both splits but, on 151 test images, is not significant at the group level.

## Asbestos type (multi-class)

The second research task is naming the material class, not only whether asbestos is present. The macro models trained on both questions (the rebuilt-split models) were scored on the 151 test images of the rebuilt split. Each candidate class name was scored by its full log-likelihood, and the asbestos / non-asbestos branch came from the binary decision whose threshold was chosen on validation, so the type result adds only the choice among types.

| Macro model (rebuilt split) | Accuracy, 4 classes | Balanced accuracy | Macro F1 | Cohen's κ | Asbestos images: type correct | Balanced accuracy, asbestos images only |
| --- | --- | --- | --- | --- | --- | --- |
| Original | 0.907 (137 / 151) | 0.776 | 0.768 | 0.840 | 67 / 76 (0.882) | 0.725 |
| With synthetic data | 0.980 (148 / 151) | 0.919 | 0.936 | 0.965 | 73 / 76 (0.961) | 0.892 |

| Class (test images) | Original | With synthetic data |
| --- | --- | --- |
| Chrysotile (63) | 57 | 62 |
| Chrysotile and Organic Fiber (6) | 4 | 5 |
| Chrysotile, Organic Fiber and Cement Sheet (7) | 4 | 6 |
| Cement Sheet (75) | 72 | 75 |

- **Synthetic data helps type identification more than detection.** Paired by near-duplicate group, the synthetic-data model is right on 9 groups the original gets wrong and wrong on 1 it gets right (McNemar exact test, p = 0.021; accuracy +0.066, 95% CI 0.017–0.116). For detection alone the same comparison gave p = 0.0625.
- **Its 3 errors are all subtype confusions** between the three chrysotile classes; it makes no detection error. Of the original model's 14 errors, 6 are detection errors and 8 are subtype confusions.
- **The two mixed classes are very small** (6 and 7 test images), so their per-class results are indicative only.

**Microscope model.** The detection model was trained on the detection question only; asked for a type, it gives almost the same answer for every image (mostly "Chrysotile Fluffy"). A second microscope model was therefore trained on the type question alone, with the same recipe (1 epoch, 4 tiles per image, class-balanced sampling over the seven classes). It was evaluated the same way as the macro models: the detection decision comes from the detection model with its validated rule, and only the five tiles the detection model scores highest in each image are scored for type.

| Microscope test images | Detection-only model | Type-trained model |
| --- | --- | --- |
| All 1,781 images, 7 classes: accuracy | 0.394 | 0.653 (95% CI 0.632–0.676) |
| Balanced accuracy | 0.300 | 0.501 |
| Cohen's κ | 0.246 | 0.536 |
| 993 asbestos images, 5 asbestos classes: accuracy | 0.383 | 0.645 |
| Balanced accuracy (chance 0.20) | 0.201 | 0.443 |
| 1,194 images in shared classes: accuracy | 0.886 | 0.830 |
| Balanced accuracy | 0.614 | 0.692 |

- **Training on type works.** On asbestos images, balanced accuracy rose from chance (0.201) to 0.443, and Cohen's κ over all classes from 0.246 to 0.536.
- **Most errors are type errors, not detection errors.** The detection decision is right on about 91% of images (0.907 with the validated rule), but the type is right on 65%, so roughly 450 images are detected correctly and given the wrong type.
- **Shared-class accuracy fell while balanced accuracy rose.** The detection-only model's 0.886 is inflated: it called nearly every asbestos image Chrysotile Fluffy, which the shared classes merge with Chrysotile Pointy into Chrysotile, the largest shared class. The type-trained model spreads its answers over all seven classes, and an amosite or crocidolite answer counts as wrong in the shared classes. Balanced accuracy is the fairer measure here.

**Longer training (3 epochs).** To test whether the short training explains the result, the microscope type model was retrained for 3 epochs with otherwise identical settings (7 h 29 min on one GPU). Validation loss went 0.199, 0.201, 0.185 over the three epochs, and the epoch-3 checkpoint was kept. Both models were scored on the same 1,781 test images, so the comparison is paired (McNemar's test by sample).

| Microscope test images | 1 epoch | 3 epochs | Change |
| --- | --- | --- | --- |
| All images, 7 classes: accuracy | 0.653 | 0.673 | +0.020 (95% CI 0.008–0.033), p = 0.002 |
| Balanced accuracy | 0.501 | 0.445 | −0.056 |
| Asbestos images, 5 classes: accuracy | 0.645 | 0.675 | +0.030 |
| Balanced accuracy (chance 0.20) | 0.443 | 0.342 | −0.101 |
| Shared classes: accuracy | 0.830 | 0.864 | +0.034 (95% CI 0.022–0.048), p = 7.0 × 10⁻⁷ |
| Shared classes: balanced accuracy | 0.692 | 0.572 | −0.120 |
| Shared classes: macro F1 | 0.679 | 0.610 | −0.069 (95% CI −0.120 to −0.023) |

| Class (test images) | Recall, 1 epoch | Recall, 3 epochs |
| --- | --- | --- |
| Chrysotile Pointy (578) | 0.609 | 0.640 |
| Chrysotile Fluffy (375) | 0.563 | 0.616 |
| Organic Fiber (564) | 0.711 | 0.725 |
| Cement Sheet (224) | 0.826 | 0.826 |
| Crocidolite (18) | 0.389 (7) | 0.111 (2) |
| Chrysotile and Organic Fiber (17) | 0.412 (7) | 0.000 (0) |
| Amosite (5) | 0.000 (0) | 0.200 (1) |

- **Longer training traded the rare types for the common ones.** Recall rose on the large classes and fell sharply on crocidolite and on chrysotile with organic fibre, which the 3-epoch model never names correctly. Accuracy counts every image, so it is dominated by the large classes and went up; balanced accuracy and macro F1 weight each class equally and went down.
- **Validation loss agrees with accuracy, not with balanced accuracy**, because the validation set is dominated by the same large classes. Choosing the checkpoint on validation loss therefore favours the common types.
- **Which model is better depends on the goal.** For the most correct images, the 3-epoch model; for naming every type, including the rare ones, the 1-epoch model. With these imbalanced classes, balanced accuracy is the better summary of type identification, and on it the 1-epoch model is ahead. The rare classes have only 5–18 test images each, so their recalls are uncertain.

**Keeping the rare classes.** Training already sampled every class equally often (inverse-frequency sampling), so each tile of a rare class was repeated many times per epoch: amosite has about 18 training images. Two further runs (type question only, up to 3 epochs, otherwise the same settings) tried to stop the model memorising those few tiles:

- **Rare-class settings:** classes sampled in proportion to the square root of their size instead of equally, and the checkpoint chosen on the validation loss averaged per class rather than per example.
- **Rare-class settings + synthetic data:** as above, plus 20 synthetic copies of every training image of amosite, crocidolite and chrysotile with organic fibre (flips, rotations, mild affine and brightness changes, blur and noise; colour hue is not changed because it is diagnostic in polarised-light microscopy).

Both runs kept their epoch-1 checkpoint: the per-class validation loss rose after the first epoch (0.406, 0.524, 0.591 and 0.440, 0.482, 0.522), consistent with the model memorising rare-class tiles when trained longer.

**Correcting for the class mix seen in training.** A model learns how often each class appeared in training and leans its answers that way. Sampling and synthetic copies change that mix, so the saved class probabilities of each image were reweighted by (target share / share in training) and the type decided again, keeping the image's asbestos / non-asbestos decision. The training share is rebuilt from each run's settings (tiles, synthetic copies, sampling weights); no test data is used. With a uniform target, every class is treated as equally likely, which is what balanced accuracy rewards. The two equal-sampling runs (1 and 3 epochs) are already uniform, so the correction leaves them unchanged.

| Microscope test images (1,781) | 1 epoch | 3 epochs | Rare-class settings, corrected | Rare + synthetic, corrected |
| --- | --- | --- | --- | --- |
| Accuracy, 7 classes | 0.653 | **0.673** | 0.623 | 0.570 |
| Balanced accuracy, 7 classes | 0.501 | 0.445 | **0.590** | 0.567 |
| Balanced accuracy, 5 asbestos classes (chance 0.20) | 0.443 | 0.342 | **0.588** | 0.546 |
| Shared classes: accuracy | 0.830 | **0.864** | 0.732 | 0.714 |
| Shared classes: balanced accuracy | 0.692 | 0.572 | 0.592 | **0.793** |

| Class (test images) | 1 epoch | 3 epochs | Rare-class settings, corrected | Rare + synthetic, corrected |
| --- | --- | --- | --- | --- |
| Amosite (5) | 0 | 1 | **3** | 2 |
| Crocidolite (18) | 7 | 2 | **12** | 8 |
| Chrysotile and Organic Fiber (17) | 7 | 0 | 6 | **13** |
| Chrysotile Fluffy (375) | 0.563 | **0.616** | 0.557 | 0.125 |
| Chrysotile Pointy (578) | 0.609 | 0.640 | 0.488 | **0.721** |
| Cement Sheet (224) | 0.826 | 0.826 | 0.670 | **0.960** |
| Organic Fiber (564) | 0.711 | 0.725 | **0.794** | 0.555 |

Rare classes are given as images correct, common classes as recall.

- **Without the correction, neither new run helped.** Uncorrected, the rare-class run reached balanced accuracy 0.409 and the synthetic-data run 0.464, both below the 1-epoch model's 0.501.
- **With it, the rare-class settings give the best microscope type model:** balanced accuracy 0.590 over seven classes and 0.588 over the five asbestos types, against 0.501 and 0.443 for the 1-epoch model. Most of the gain is on the rare types (amosite 3 of 5, crocidolite 12 of 18), paid for with lower accuracy on the common ones. Sampling rare classes less aggressively and correcting the decision afterwards works better than forcing equal sampling during training.
- **Synthetic copies helped the class they targeted but damaged a similar one.** Chrysotile with organic fibre rose to 13 of 17, but Chrysotile Fluffy collapsed from 0.56 to 0.13 and the correction did not restore it. Of the 375 fluffy chrysotile images, the uncorrected synthetic-data model called 118 chrysotile with organic fibre (15 for the 1-epoch model) and 172 chrysotile pointy (92 before). Since the correction removes the effect of the class mix, the copies changed what the model treats as fluffy chrysotile, not only how often it answers it. Its high shared-class balanced accuracy (0.793) comes from the shared classes merging fluffy and pointy chrysotile, which hides this error.
- **The rare classes have 5–18 test images,** so their counts are uncertain: 12 of 18 has a 95% interval of roughly 0.41–0.87. The direction is consistent across amosite and crocidolite, but a larger test set of rare types would be needed to confirm it.

**Microscope vs macro, type.** The test sets contain different specimens, so the comparison is unpaired (Fisher's exact test, computed from the counts above).

|  | Microscope, 1 epoch | Microscope, 3 epochs | Microscope, rare-class settings, corrected | Macro (synthetic data, rebuilt split) |
| --- | --- | --- | --- | --- |
| Shared classes: accuracy | 0.830 (991/1,194), p = 2.6 × 10⁻⁸ | 0.864 (1,032/1,194), p = 4.3 × 10⁻⁶ | 0.732 (874/1,194), p = 2.5 × 10⁻¹⁵ | 0.980 (148/151) |
| Shared classes: balanced accuracy | 0.692 | 0.572 | 0.592 | 0.943 |
| Balanced accuracy, own classes | 0.501 (7 classes) | 0.445 | 0.590 | 0.919 (4 classes) |

p values compare each microscope model with the macro model.

- The shared-class comparison is the main one: both sides use the same three classes (Chrysotile, Chrysotile and Organic Fiber, Cement Sheet).
- The own-class comparison is indicative only: the microscope model chooses among five asbestos types (including amosite and crocidolite), the macro model among three chrysotile types.
- Training length does not explain the gap. Tripling the microscope model's training raised its shared-class accuracy only from 0.830 to 0.864, still far below the macro model's 0.980, and lowered its balanced accuracy. The comparison still mixes image type with data (the macro model also had synthetic data), so it shows what these models reach, not that macro images are inherently better for naming the type.

## Limitations

The headline numbers are a single training run per model, and seven factors could bias them or the comparison between image types.

| Issue | Effect on the results | How to address it |
| --- | --- | --- |
| Splits per image, not per specimen | Several microscope fields of the same sample may sit in both training and test, which would inflate the microscope score | Rebuild the splits with a specimen id (`--sample_id_regex`) and re-run |
| Unequal training | 11,818 microscope images, 1 epoch on 4 tiles each, vs 995 macro images for up to 3 epochs. The gap mixes image type with data size and training length | Match training budgets, or train one model on both image types and test it on each |
| Small macro test set (145 images) | Wide intervals; p = 0.094 means no evidence of a difference, not evidence of equality | More macro images, or cross-validation so every image is tested once |
| Different class mixes | Microscope negatives include organic fibre and cement sheet; macro negatives are cement sheet only | Report results per class; compare on shared classes |
| No offline augmentation | Only on-the-fly flips were used; results may improve with the box-preserving augmentation | Done for the macro model (see Macro model with synthetic training data); not yet for the microscope model |
| Labels from boxes | 1,342 images had boxes of several classes; image labels follow the largest asbestos box | Check a sample of multi-class images by hand |
| Boxes do not mark every fibre | Tiles counted as empty often hold fibres (seen in the overlay images), so the localisation numbers understate the model; training was unaffected because unboxed tiles of asbestos images were left out | Box every fibre in a sample of test images and repeat the localisation check on those |

Asbestos type identification (the second research task) has been evaluated for both image types, with separate microscope models trained on the type question. The rare microscope types (amosite, crocidolite, chrysotile with organic fibre) have only 26–117 images in total and 5–18 test images each, so their results are uncertain; the best microscope model needs its decisions corrected for the class mix it was trained on, and synthetic copies of rare classes can blur them with similar common classes. More real images of the rare types are the most direct way to improve them.
