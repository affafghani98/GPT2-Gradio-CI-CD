import gradio as gr
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')# Loading pre-trained model
# Define the prediction function
def generate_text(prompt, max_length=50, num_return_sequences=1):
    outputs = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return [output['generated_text'] for output in outputs]
iface = gr.Interface(# Gradio interface
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Slider(minimum=10, maximum=100, value=50, label="Max Length"),
        gr.Slider(minimum=1, maximum=5, value=1, label="Number of Outputs")
    ],
    outputs=[
        gr.Textbox(label="Generated Text")
    ],
    title="GPT-2 Text Generator",
    description="Enter a prompt to generate text using the GPT-2 model.",
    examples=[
        ["Once upon a time"],
        ["In a galaxy far, far away"],
        ["The meaning of life is"]
    ]
)

if __name__ == "__main__":
    iface.launch()
