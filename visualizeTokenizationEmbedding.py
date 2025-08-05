# tokenization using the Hugging Face Transformers library
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokens = tokenizer.tokenize("A young man called Ziang was walking through the forest...")

# Print the tokens
print(tokens)

# embedding and processing with a transformer model
from transformers import AutoModel

model = AutoModel.from_pretrained("distilgpt2")
inputs = tokenizer("A young man called Ziang was walking through the forest...", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

# visualization of token embeddings
import matplotlib.pyplot as plt
plt.imshow(last_hidden_states.detach().numpy()[0], cmap='viridis')
plt.colorbar()
plt.title("Token Embeddings Visualization")
plt.show()

