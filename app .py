import os
import tensorflow as tf
from tensorflow.image import resize
from tensorflow.keras.applications.efficientnet import preprocess_input
import gradio as gr

# -----------------------------
# Costanti
# -----------------------------
MODEL_PATH = "effnet_model.keras"  # modello pre-addestrato
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Benign", "Malignant"]
THRESHOLD = 0.5  # soglia per considerare 'Malignant'

# -----------------------------
# Carica modello
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Place the .keras file in the same folder.")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# -----------------------------
# Funzione di predizione
# -----------------------------
def predict_image(img):
    """
    img: PIL.Image
    """
    # Resize e preprocess
    img = resize(tf.convert_to_tensor(img), IMG_SIZE)
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predizione
    preds = model.predict(img)[0]  # array: [prob_benign, prob_malignant]
    prob_benign = float(preds[0])
    prob_malignant = float(preds[1])

    # Applica threshold
    if prob_malignant > THRESHOLD:
        predicted_class = "Malignant"
    else:
        predicted_class = "Benign"

    # Formatta output tutto insieme
    output_text = (
        f"Predicted Class: {predicted_class}\n\n"
        f"Probabilities:\n"
        f" - Benign: {prob_benign*100:.2f}%\n"
        f" - Malignant: {prob_malignant*100:.2f}%\n\n"
        f"⚠️ Warning: This model is for educational purposes only and does not replace a professional medical diagnosis."
    )

    return output_text

# -----------------------------
# Interfaccia Gradio
# -----------------------------
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(lines=8),
    title="Skin Cancer Classifier",
    description=(
        "Upload an image of a skin lesion to classify it as benign or malignant. "
        "⚠️ Warning: this model is for educational and research purposes only. "
        "It does not in any way replace a professional medical diagnosis."
    )
)

# Avvia l'app
interface.launch()

