# patch_match
Matching patches of size 512x512px and stride 25px across orthorectified maps registered on various different seasons and times of days

# To train your own models:

1. pip install -r requirements.txt

2. python3 generate_patches_test.py

3. python3 generate_patches_train.py

4. python3 train_xxxx.py here xxxx can be siamese, classifier, classifier_arcface or triplet_loss

# For testing
To run patch matching across all patches of drone with all patches of maps using all models available.

1. Download the models from the link : https://iiitaphyd-my.sharepoint.com/:f:/g/personal/unni_krishnan_alumni_iiit_ac_in/EjLRjvZWin9HlJ-YqcjdijcB_24gQIKGyrAZtzz4QfA8dA?e=LFSWZl

2. Extract it into root of this repo.

3. python3 match_all_vs_all.py
