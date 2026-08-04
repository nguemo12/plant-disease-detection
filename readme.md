Shared-Backbone Multi-Head CNN for crop-specific plant disease diagnosis

A custom shared conv backbone extracts general leaf texture/lesion features across all crops. Each crop gets its own classification head on top of that shared feature vector. Training routes each image to its crop's head by the known label (masked multi-task loss); the Streamlit app routes by the crop the user selects.

Pipeline, in order
build_dataset_index.py <dataset_root> --out_dir work Parses your <Crop>___<Disease> folders into dataset_index.csv, crop_to_idx.json, disease_to_idx_per_crop.json.
audit_class_balance.py (run from work/) Reports per-crop and per-class image counts, flags thin classes/crops. Do this before touching architecture — it decides how much augmentation or oversampling you need, and whether any crop should be dropped for now.
make_splits.py (run from work/) Adds a stratified split column (train/val/test) to dataset_index.csv, stratified by (crop, disease) so every class is represented in every split. Classes with under 3 images go entirely to train, with a warning — the audit step should already have flagged these.
train.py Trains the shared backbone + all crop heads jointly. Masked multi-task loss: each batch is grouped by crop, only the matching head runs on each group, and per-sample cross-entropy is summed across groups and divided by batch size — every image contributes exactly once, through its own head, and the backbone gets gradient signal from every crop each step. Saves the best checkpoint (by validation macro-F1) to best_model.pt. Key args: --epochs, --batch_size, --lr, --patience (early stop).
evaluate.py Per-crop classification report and confusion matrix on the test split, plus a summary table (crop, n, accuracy, macro-F1). Deliberately does not report one blended number — a strong head can hide a failing one.
app.py — streamlit run app.py User picks the crop, uploads a leaf photo, gets the top predictions from that crop's head with confidence scores. Run this from the same directory as best_model.pt.
Architecture notes
Backbone: 4 conv blocks (32→64→128→256 channels), each block = conv-BN-ReLU ×2 + maxpool, dropout increasing 0.1→0.4, global average pooling. This is a baseline, not a fixed choice — after your first training run, check the train/val gap:
both plateau low, gap small → underfitting, add a block or widen channels
train pulls well ahead of val → overfitting, add regularization (dropout, augmentation, weight decay) rather than removing depth
Heads: small MLP per crop (Linear → ReLU → Dropout → Linear), sized to that crop's number of disease classes.
Augmentation: geometric (flip, rotation) plus mild brightness/contrast only — color/stripe pattern is diagnostic for several diseases, so heavy hue/saturation jitter is deliberately avoided.