import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from datasets import load_dataset
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model names and paths
model_name = "runwayml/stable-diffusion-v1-5"
output_dir = "./sd-lora-output"
os.makedirs(output_dir, exist_ok=True)

# Load tokenizer and models
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

# LoRA configuration
lora_config = LoraConfig(
    r=4,  # Rank of the LoRA approximation
    lora_alpha=16,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Target modules to apply LoRA
    lora_dropout=0.0,
    bias="none",
    task_type="UNET",
)

# Apply LoRA to the UNet model
unet = get_peft_model(unet, lora_config)
unet = unet.to(device)

# Freeze all parameters except LoRA parameters
for param in unet.parameters():
    param.requires_grad = False

for param in unet.parameters():
    if param.requires_grad:
        print("Error: There should be no trainable parameters.")

for name, module in unet.named_modules():
    if "lora_" in name:
        for param in module.parameters():
            param.requires_grad = True

# Collect parameters to optimize
params_to_optimize = [param for param in unet.parameters() if param.requires_grad]

# Check if there are parameters to optimize
if not params_to_optimize:
    raise ValueError("No parameters to optimize. Check if LoRA layers are correctly applied.")

# Define optimizer
optimizer = AdamW(params_to_optimize, lr=1e-4)

# Initialize noise scheduler
noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

# Load and preprocess dataset
# Replace 'path_to_your_dataset' with your actual dataset path
dataset = load_dataset('imagefolder', data_dir='./training_set')

# Preprocessing function
def preprocess(example):
    image = example['image']
    image = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    example['pixel_values'] = preprocess(image)
    example['text'] = example['text'] if 'text' in example else ''

    return example

dataset = dataset.map(preprocess, remove_columns=['image'])
def collate_fn(examples):
    # for example in examples:
    #     pixel_values = torch.stack([example['pixel_values']])
    #     # Überprüfen des Typs von 'pixel_values'
    #     print(type(pixel_values))
    pixel_values = torch.stack([torch.tensor(example['pixel_values']) for example in examples])

    texts = [example['text'] for example in examples]
    return {'pixel_values': pixel_values, 'text': texts}

dataloader = DataLoader(dataset['train'], batch_size=1, shuffle=True, collate_fn=collate_fn)


# Training loop
num_epochs = 1
global_step = 0
gradient_accumulation_steps = 1
max_train_steps = num_epochs * len(dataloader)

progress_bar = tqdm(range(max_train_steps), desc="Training")

unet.train()
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            # Encode images to latent space
            pixel_values = batch['pixel_values'].to(device)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise and add to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Tokenize text
            text = batch.get('text', [""])[0]
            inputs = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids = inputs.input_ids.to(device)

            # Encode text
            encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        global_step += 1

# Save LoRA weights
unet.save_pretrained(output_dir)
print(f"LoRA weights saved to {output_dir}")

# --- Image generation with the fine-tuned LoRA model ---

# Load LoRA weights
from peft import PeftModel

unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
unet = PeftModel.from_pretrained(unet, output_dir)
unet = unet.to(device)

# Create pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    model_name,
    unet=unet,
    text_encoder=text_encoder,
    vae=vae,
    tokenizer=tokenizer,
).to(device)

# Define the prompt for image generation
prompt = "A cute squirrel with a white karate suite"

# Generate the image
with torch.no_grad():
    images = pipeline(prompt, num_inference_steps=5, guidance_scale=7.5, height=512,
        width=512).images

# Save the generated image
output_image_path = os.path.join(output_dir, "generated_image.png")
images[0].save(output_image_path)
print(f"Image generated and saved to {output_image_path}")
