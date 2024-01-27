from datasets import load_dataset
# Load the Falah/Alzheimer_MRI dataset
dataset = load_dataset('Falah/Alzheimer_MRI')

print(dataset)

from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

configs = ViTConfig(
    image_size=224,
    )

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=4)

