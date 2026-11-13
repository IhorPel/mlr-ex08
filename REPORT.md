# Task 4 Report

This draft uses the saved experiment results. A real phone-camera test, a trial in the course camera app and confirmation that the historical dataset is accepted are still pending.

## Setup

The data came from the historical course ZIP supplied for this run. There were 700 training images and 120 validation images, covering apple, cherry, chestnut and maple. These were not the current group's photographs.

The baseline was the Task 3 CNN with four convolution blocks, trained for 15 epochs using Adam, learning rate 0.001 and batches of 32. The first comparisons used seed 0. Each single-change experiment kept the other baseline settings fixed. The transfer-learning experiments used MobileNet and are shown separately from the scratch CNN changes.

Every variant was also measured on a camera-like validation set. It was made once by cropping, rotating and changing the colours of the validation images, using seed 123. Every model saw the same saved versions. This checks sensitivity to those changes, but it is not a separate collection of real camera photographs.

## Results

The table shows seed 0. Accuracy is given as a percentage. Most runs used the final epoch after 15 epochs. The final combined transfer model trained for up to 30 epochs and kept its best clean-validation checkpoint, which was epoch 11 for seed 0.

| Model or change | Clean validation | Camera-like validation |
| --- | ---: | ---: |
| Baseline CNN | 84.17 | 63.33 |
| Geometry augmentation | 87.50 | 70.00 |
| Batch normalization | 83.33 | 70.00 |
| Batch normalization and geometry | 79.17 | 72.50 |
| Colour augmentation | 84.17 | 70.00 |
| Dropout | 82.50 | 64.17 |
| Global average pooling | 66.67 | 58.33 |
| AdamW | 82.50 | 64.17 |
| Label smoothing | 80.83 | 63.33 |
| Best epoch | 84.17 | 63.33 |
| Cosine scheduler | 81.67 | 62.50 |
| Frozen MobileNet | 85.83 | 80.83 |
| Fine-tuned MobileNet, also the Task 2 demo | 89.17 | 74.17 |
| Fine-tuned MobileNet with geometry | 89.17 | 88.33 |
| Fine-tuned MobileNet with cosine scheduler | 92.50 | 87.50 |
| CNN with geometry and colour | 81.67 | 75.83 |
| MobileNet with geometry, colour, cosine scheduler and best checkpoint | 90.83 | 90.83 |

The full 22 rows, including the repeated seeds, are in results.csv. The Task 2 demo and fine-tuned MobileNet row use the same trained model. They are not two independent runs.

## What helped and what did not

Geometry and colour augmentation each improved the baseline's camera-like accuracy from 63.33% to 70.00%. Batch normalization also reached 70.00%, although its clean accuracy was slightly lower. Combining batch normalization with geometry improved the camera-like result further, but reduced clean accuracy. This shows that a higher clean score does not always mean a more robust model.

Dropout and AdamW each improved the camera-like result by only one image. Their clean scores were lower than the baseline, so this is not convincing evidence of an improvement. Label smoothing and the scratch CNN's cosine scheduler did not help here. Global average pooling made both scores worse. Keeping the best epoch made no difference to the seed-0 baseline because its best epoch was already the last one.

Transfer learning gave the strongest results. Adding geometry to fine-tuned MobileNet raised camera-like accuracy from 74.17% to 88.33%, with the same clean accuracy. Frozen MobileNet was less accurate on clean images than fine-tuning, but better on the camera-like set. The frozen run also keeps feature batch normalization and dropout in evaluation mode, so it represents the whole frozen-feature workflow.

The final transfer combination performed well on both sets. It changes several settings together and uses more epochs, so its improvement cannot be assigned to one component alone.

## How much is noise?

The baseline and selected transfer combination were repeated with seeds 0, 1 and 2. The values below are means and standard deviations in percentage points, using ddof=0.

| Model family | Clean mean | Clean SD | Camera-like mean | Camera-like SD |
| --- | ---: | ---: | ---: | ---: |
| Baseline CNN | 81.94 | 3.14 | 63.33 | 1.36 |
| Selected transfer combination | 91.94 | 1.04 | 91.94 | 0.79 |

The mean camera-like improvement is 28.61 percentage points, much larger than the observed variation between these seeds. However, three seeds are not a formal significance test. Most other variants were run with only one seed, so their smaller differences remain uncertain.

One validation image changes accuracy by about 0.83 percentage points. The clean and camera-like sets also share the same original images. Repeatedly comparing models and choosing checkpoints on these sets can make the best observed result optimistic. There is no untouched final test set in this experiment.

## Model for the phone

My choice from these measurements would be the augmented MobileNet with the cosine scheduler and best-validation checkpoint. Its camera-like mean was the highest among the repeated families. The exported model uses seed 0 rather than choosing whichever random seed scored best. It is available as best_leaf_model.onnx and as the Task 4 option on the page.

The browser check confirms that its probabilities match the notebook to three decimal places, and photo upload works. This does not establish how well it recognizes a live leaf on a particular phone. Testing it on a physical phone and in the course camera app is still pending.

The results support this choice for the supplied historical data, but they do not establish performance on the current group's photos, other species or different camera conditions. The teacher still needs to confirm whether the dataset substitution is acceptable.
