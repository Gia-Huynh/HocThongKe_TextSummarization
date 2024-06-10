from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the path to your saved model
checkpoint = "./my_awesome_billsum_model/checkpoint-500"

# Load the trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

# Define your input text
input_text = "You can't just replace the processor on the UNO with something better. The 328P is pretty much the best MCU there is in the pinout for that board.Instead you can replace the whole UNO with something more powerful. My board of choice for AI work would have to be one based on the Kendryte K210 chip, such as the Maixduino (which is helpfully in an UNO footprint). The K210 is a dual-core 64-bit 400MHz RISC-V CPU with embedded neural network co-processor. On the Maixduino it's also coupled with an ESP32 for WiFi/Bluetooth. That's 2x 400MHz 64-bit cores, 2x 32-bit 240MHz cores, one low-power FSM core, and a neural network core all on one board."

# Tokenize the input text
inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary (you can tweak the parameters like max_length and num_beams as needed)
summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode the generated summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print("Summary:", summary)
