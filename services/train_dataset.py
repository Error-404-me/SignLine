# # patch_skipped_labels.py
# import csv, cv2, mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from pathlib import Path

# SIGNS_DIR  = Path("speak_sign_app/assets/signs")
# OUTPUT_CSV = Path("speak_sign_app/data/fsl_samples.csv")
# TASK_MODEL = Path("speak_sign_app/assets/models/hand_landmarker.task")

# # Only reprocess these labels with relaxed confidence
# PROBLEM_LABELS = {"M", "N", "O", "Q", "S"}

# # Load already-saved rows so we don't duplicate
# saved_rows: list[dict] = []
# saved_counts: dict[str, int] = {}
# with OUTPUT_CSV.open(encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     fieldnames = reader.fieldnames
#     for row in reader:
#         saved_rows.append(row)
#         saved_counts[row["label"]] = saved_counts.get(row["label"], 0) + 1

# # Build relaxed detector
# base_options = python.BaseOptions(model_asset_path=str(TASK_MODEL))
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     num_hands=1,
#     min_hand_detection_confidence=0.2,   # relaxed from 0.5
#     min_hand_presence_confidence=0.2,
#     min_tracking_confidence=0.2,
# )
# detector = vision.HandLandmarker.create_from_options(options)

# new_rows = []
# skipped = 0

# for label in sorted(PROBLEM_LABELS):
#     label_dir = SIGNS_DIR / label
#     if not label_dir.exists():
#         continue

#     already_saved = saved_counts.get(label, 0)
#     images = list(label_dir.glob("*.[jp][pn]g"))
#     print(f"Processing {label!r}: {len(images)} images, {already_saved} already saved...")

#     # Track which images are already in the CSV by index isn't possible,
#     # so re-extract all and replace existing rows for this label
#     label_rows = []
#     for img_path in images:
#         img = cv2.imread(str(img_path))
#         if img is None:
#             skipped += 1
#             continue

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#         )
#         result = detector.detect(mp_image)
#         if not result.hand_landmarks:
#             skipped += 1
#             continue

#         lm = result.hand_landmarks[0]
#         row = {"label": label}
#         for i, point in enumerate(lm):
#             row[f"f{i*3}"]   = round(point.x, 6)
#             row[f"f{i*3+1}"] = round(point.y, 6)
#             row[f"f{i*3+2}"] = round(point.z, 6)
#         label_rows.append(row)

#     print(f"  → Recovered {len(label_rows)} / {len(images)} (skipped {len(images) - len(label_rows)})")
#     new_rows.extend(label_rows)

# detector.close()

# # Merge: keep non-problem labels, replace problem labels with new rows
# merged = [r for r in saved_rows if r["label"] not in PROBLEM_LABELS] + new_rows

# with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(merged)

# print(f"\nDone. Total rows: {len(merged)} | Still skipped: {skipped}")
# print(f"CSV updated: {OUTPUT_CSV}")


# import csv
# import pickle
# from pathlib import Path

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# dataset = Path("speak_sign_app/data/fsl_samples.csv")
# model_path = Path("speak_sign_app/assets/models/fsl_rf.pkl")

# X, y = [], []
# with dataset.open(newline="", encoding="utf-8") as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         y.append(row["label"])
#         X.append([float(row[f"f{i}"]) for i in range(63)])

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
# clf.fit(X_train, y_train)
# print(classification_report(y_test, clf.predict(X_test)))

# model_path.parent.mkdir(parents=True, exist_ok=True)
# with model_path.open("wb") as file:
#     pickle.dump(clf, file)
# print(f"Saved {model_path}")

# run this standalone — save as check_npy.py
# import numpy as np
# from pathlib import Path

# sample = np.load(
#     Path("speak_sign_app/assets/FSL_img_dataset_MediaPipe_24classes.npy"),
#     allow_pickle=True,
# )
# print("Shape:", sample.shape)
# print("Dtype:", sample.dtype)
# print("First item:", sample[0] if sample.ndim > 0 else sample)

# train_combined.py
# import csv, pickle
# import numpy as np
# from pathlib import Path
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# NPY_FILE  = Path("speak_sign_app/assets/FSL_img_dataset_MediaPipe_24classes.npy")
# CSV_FILE  = Path("speak_sign_app/data/fsl_samples.csv")
# MODEL_OUT = Path("speak_sign_app/assets/models/fsl_rf.pkl")

# X, y = [], []

# # ── Source 1: .npy dataset (24 classes, pre-extracted landmarks) ──────────
# item    = np.load(NPY_FILE, allow_pickle=True).item()
# samples = item["data"]
# labels  = item["target"]

# npy_count = 0
# for sample, label in zip(samples, labels):
#     if sample is None:
#         continue
#     if not isinstance(sample, np.ndarray) or sample.shape != (63,):
#         continue
#     X.append(sample.tolist())
#     y.append(str(label))
#     npy_count += 1

# print(f"NPY samples loaded : {npy_count}")

# # ── Source 2: your own CSV dataset (26 classes, your camera images) ───────
# csv_count = 0
# if CSV_FILE.exists():
#     with CSV_FILE.open(encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             X.append([float(row[f"f{i}"]) for i in range(63)])
#             y.append(row["label"].strip().upper())
#             csv_count += 1
#     print(f"CSV samples loaded : {csv_count}")
# else:
#     print(f"CSV not found — skipping: {CSV_FILE}")

# print(f"\nCombined total : {len(X)}")
# print(f"Labels         : {sorted(set(y))}")

# # ── Train ─────────────────────────────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# clf = RandomForestClassifier(
#     n_estimators=250,
#     random_state=42,
#     class_weight="balanced",   # handles uneven class sizes between sources
# )
# clf.fit(X_train, y_train)
# print("\n", classification_report(y_test, clf.predict(X_test)))

# # ── Save ──────────────────────────────────────────────────────────────────
# MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
# with MODEL_OUT.open("wb") as f:
#     pickle.dump(clf, f)
# print(f"Saved → {MODEL_OUT}")

# train_mlp.py
import csv, pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

NPY_FILE  = Path("speak_sign_app/assets/FSL_img_dataset_MediaPipe_24classes.npy")
CSV_FILE  = Path("speak_sign_app/data/fsl_samples.csv")
MODEL_OUT = Path("speak_sign_app/assets/models/fsl_mlp.h5")

# ── Load both datasets (same as train_combined.py) ────────────────────────
X, y = [], []

item = np.load(NPY_FILE, allow_pickle=True).item()
for sample, label in zip(item["data"], item["target"]):
    if sample is None or not isinstance(sample, np.ndarray) or sample.shape != (63,):
        continue
    X.append(sample.tolist())
    y.append(str(label))

if CSV_FILE.exists():
    with CSV_FILE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append([float(row[f"f{i}"]) for i in range(63)])
            y.append(row["label"].strip().upper())

X = np.array(X, dtype=np.float32)
print(f"Total samples : {len(X)}")
print(f"Labels        : {sorted(set(y))}")

# ── Encode labels to integers ─────────────────────────────────────────────
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Classes       : {list(encoder.classes_)}")

# Save encoder classes alongside the model
encoder_out = MODEL_OUT.parent / "fsl_mlp_classes.pkl"
with encoder_out.open("wb") as f:
    pickle.dump(encoder.classes_, f)
print(f"Classes saved → {encoder_out}")

# ── Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── Build MLP with Keras ──────────────────────────────────────────────────
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(63,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ],
)

# ── Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ── Save ──────────────────────────────────────────────────────────────────
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_OUT)
print(f"Saved → {MODEL_OUT}")