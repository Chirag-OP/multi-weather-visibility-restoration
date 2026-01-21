import tensorflow as tf

MODEL_PATH = "my_model"   # folder containing saved_model.pb

print("=== START ===")

print("Loading SavedModel...")
model = tf.saved_model.load(MODEL_PATH)

print("\nAvailable signatures:")
for key in model.signatures:
    print(" -", key)

print("\nDetailed serving_default signature:")
infer = model.signatures["serving_default"]
print(infer)

print("=== END ===")
