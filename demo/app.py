import gradio as gr
from transformers import pipeline
import os

REPO   = os.getenv("HF_REPO_ID", "Backened/sarcasm-model")
LABELS = {"LABEL_0": "Not Sarcastic", "LABEL_1": "Sarcastic"}
CUES   = [
    "oh great","oh wow","oh sure","just what","so helpful","love how",
    "clearly","obviously","yeah right","only took","amazing","wonderful",
    "fantastic","totally","absolutely","of course","sure","as if",
]

clf = pipeline("text-classification", model=REPO, device=-1)

def predict(text: str, threshold: float):
    if not text.strip():
        return "—", 0.0, "—", "—"

    out   = clf(text[:512])[0]
    label = LABELS.get(out["label"], out["label"])
    conf  = round(float(out["score"]), 4)

    if conf < threshold:
        display = f"Uncertain  (confidence {conf:.0%} < threshold {threshold:.0%})"
    else:
        display = label

    signals = [f'"{c}"' for c in CUES if c in text.lower()]
    sig_txt = "Detected: " + ", ".join(signals[:4]) if signals else "No common sarcasm cues"

    bar = f"{conf:.0%} confident it is {label.lower()}"
    return display, conf, sig_txt, bar


examples = [
    ["Oh great, another Monday. Just what I needed.", 0.65],
    ["I really enjoyed this product, works perfectly!", 0.65],
    ["Sure, because that always works out so well.", 0.65],
    ["Thank you for the quick response, very helpful!", 0.65],
    ["Oh wow, my flight got cancelled again. Loving this airline.", 0.65],
    ["The team did an excellent job, really proud of everyone.", 0.65],
]

with gr.Blocks(title="Sarcasm Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Sarcasm Detector
        **Fine-tuned DistilBERT** trained on 120k+ samples across Reddit, Twitter & news headlines.
        F1 score: **0.872** · Accuracy: **85.3%**

        > Try the examples below or paste your own text.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(
                label="Input text",
                placeholder="Type something sarcastic (or not)...",
                lines=3,
            )
            threshold = gr.Slider(
                minimum=0.5, maximum=0.95, value=0.65, step=0.05,
                label="Confidence threshold — below this returns 'Uncertain'",
            )
            btn = gr.Button("Detect", variant="primary")

        with gr.Column(scale=2):
            label_out  = gr.Textbox(label="Prediction")
            conf_out   = gr.Number(label="Confidence score")
            signal_out = gr.Textbox(label="Sarcasm signals found")
            bar_out    = gr.Textbox(label="Summary")

    gr.Examples(examples=examples, inputs=[text_in, threshold])

    btn.click(
        fn=predict,
        inputs=[text_in, threshold],
        outputs=[label_out, conf_out, signal_out, bar_out],
    )
    text_in.submit(
        fn=predict,
        inputs=[text_in, threshold],
        outputs=[label_out, conf_out, signal_out, bar_out],
    )

    gr.Markdown(
        """
        ---
        **Model:** `Backened/sarcasm-model` on HuggingFace Hub  
        **Code:** [GitHub](https://github.com/YOUR_USERNAME/sarcasm-detector)  
        **API:** Deployed via Render.com
        """
    )

if __name__ == "__main__":
    demo.launch()
