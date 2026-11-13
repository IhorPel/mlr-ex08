# Generated experiment evidence

This file contains measured results to inspect. It is not the student's REVIEW.md or REPORT.md.

## Dataset

The user supplied the historical archive instead of the current-group Drive ZIP. There are 175 training and 30 validation images per class, 700 and 120 total. These results do not establish performance on the current group's photographs. Camera-like validation contains one fixed crop, rotation and colour change of each of the same 120 validation images, with seed 123. It is a stress test, not an independent camera dataset.

## Task 1 optional exercises

Batch size 1 with learning rate 0.01 gives final half-MSE 0.056877. Full-batch gradient descent at learning rate 0.1 reaches half-MSE below 0.0569 after 214 epochs. For 300 epochs, split seeds 0, 1 and 2 give best validation MSE 0.0322, 0.1449 and 0.0322, at epochs 271, 201 and 120. Dropout 0.2 on split seed 0 gives 0.0344 at epoch 157. The best epoch is no longer always the last, and a different split can change the result substantially. Dropout did not improve this particular split. Exact measured values are in evidence/task1_optional.json.

## Task 2

The demonstration's clean accuracy is 0.8917, and its camera-like accuracy is 0.7417. The most-confused species pair is cherry/maple, with 6 errors across both directions. The saved ONNX check records agreement with PyTorch. Browser parity and a real phone check are separate verification steps.

## Task 3

| Variant | Seed | Epoch | Clean validation | Camera-like validation |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 15 | 0.8417 | 0.6333 |
| geometry | 0 | 15 | 0.8750 | 0.7000 |
| batchnorm | 0 | 15 | 0.8333 | 0.7000 |
| bn_geometry | 0 | 15 | 0.7917 | 0.7250 |

## Task 4: all measured variants

| Variant | Seed | Epoch | Clean validation | Camera-like validation |
| --- | ---: | ---: | ---: | ---: |
| demo_mobilenet | 0 | 15 | 0.8917 | 0.7417 |
| baseline | 0 | 15 | 0.8417 | 0.6333 |
| geometry | 0 | 15 | 0.8750 | 0.7000 |
| batchnorm | 0 | 15 | 0.8333 | 0.7000 |
| bn_geometry | 0 | 15 | 0.7917 | 0.7250 |
| colour | 0 | 15 | 0.8417 | 0.7000 |
| dropout | 0 | 15 | 0.8250 | 0.6417 |
| global_pool | 0 | 15 | 0.6667 | 0.5833 |
| adamw | 0 | 15 | 0.8250 | 0.6417 |
| label_smoothing | 0 | 15 | 0.8083 | 0.6333 |
| best_epoch | 0 | 15 | 0.8417 | 0.6333 |
| cosine | 0 | 15 | 0.8167 | 0.6250 |
| transfer_frozen | 0 | 15 | 0.8583 | 0.8083 |
| transfer_fine | 0 | 15 | 0.8917 | 0.7417 |
| transfer_geometry | 0 | 15 | 0.8917 | 0.8833 |
| transfer_cosine | 0 | 15 | 0.9250 | 0.8750 |
| selected_scratch_combination | 0 | 15 | 0.8167 | 0.7583 |
| transfer_aug_cosine_30 | 0 | 11 | 0.9083 | 0.9083 |
| baseline | 1 | 15 | 0.7750 | 0.6500 |
| baseline | 2 | 15 | 0.8417 | 0.6167 |
| transfer_aug_cosine_30 | 1 | 17 | 0.9333 | 0.9250 |
| transfer_aug_cosine_30 | 2 | 25 | 0.9167 | 0.9250 |

The CNN baseline is the notebook's four-block model trained for 15 epochs with Adam at 0.001. Single-change rows retain that baseline's settings. Frozen/fine-tuned MobileNet is a transfer-learning comparison; transfer geometry and transfer cosine are each one change relative to fine-tuned MobileNet. Combination rows change multiple settings and do not isolate their individual effects. The fine-tuned transfer seed-0 result reuses the identical Task 2 model, and its recorded training time is zero to make the reuse explicit.

## Seed variability

| Repeated family | Seeds | Clean mean / SD | Camera-like mean / SD |
| --- | --- | ---: | ---: |
| baseline | 0, 1, 2 | 0.8194 / 0.0314 | 0.6333 / 0.0136 |
| transfer_aug_cosine_30 | 0, 1, 2 | 0.9194 / 0.0104 | 0.9194 / 0.0079 |

The exported Task 4 phone candidate is **transfer_aug_cosine_30**, using seed 0. The family was chosen by mean camera-like accuracy between the repeated baseline and the initially best candidate. Seed 0 is used for export without selecting the best random seed. Other variants have only one seed, so their ranking remains exploratory. Standard deviations across three seeds describe initialization/training variability; they are not confidence intervals or a formal significance test.

One validation image changes accuracy by 1/120 = **0.833 percentage points**. The two validation sets share the same source images, so they are not independent measurements. Repeatedly selecting variants and epochs on them creates validation-selection optimism. No untouched final test set is available.

## Runtime

Recorded training wall time across actual runs: 5274.5 seconds. Per-run values are in evidence/runtimes.json. They include interruptions and machine load, so they should not be treated as hardware benchmarks. A zero value denotes reuse, not a measured zero-cost training run.

## Remaining student checks

Open the deployed URL on an actual Android phone or iPhone, test the rear camera, and take the required screenshot yourself. Confirm check_image.jpg probabilities in the browser. Try the Task 4 candidate in the course camera app if required. Write REVIEW.md and REPORT.md in your own words after checking the code and judging the results. A desktop-browser test cannot establish physical iPhone camera behavior.

## Deployed browser check

Both models passed the live page's check to three decimal places. Maximum absolute differences from notebook probabilities were 6.645e-07 for the demo and 2.385e-08 for the candidate. An actual file-chooser upload of check_image.jpg succeeded. The phone-width layout had no horizontal overflow. The QR payload matches the live Pages URL. Evidence is in evidence/browser_check.json and browser_check.jpg. These checks do not substitute for the student's physical-phone camera test and screenshot.
