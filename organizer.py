from model_builder import create_effnetb2_model
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
import os
import argparse


def get_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    return class_names


def load_weights(weights_path, model):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()


def get_prediction(input_path, file, model, transform, class_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = Image.open(f'{input_path}{file}')
    img_transformed = transform(img).unsqueeze(dim=0)
    img_to_device = img_transformed.to(device)
    img_pred = model(img_to_device)
    img_pred_probs = torch.softmax(img_pred, dim=1)
    max_value, max_index = img_pred_probs.max(dim=1)
    predicted_prob = max_value.item()
    predicted_class = class_names[max_index]

    return img, predicted_prob, predicted_class


def move_file(input_path, output_path, img, predicted_prob, predicted_class, file, remove_file=False):
    if predicted_prob < 0.35:
        dir_name = 'unknown'
    else:
        dir_name = predicted_class

    output_dir = os.path.join(output_path, dir_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    files_in_dir = os.listdir(output_dir)
    extension = os.path.splitext(file)[-1]

    if files_in_dir:
        last_file = max(files_in_dir, key=lambda f: int(f.split(dir_name)[-1].split('.')[0]))
        last_number = int(last_file.split(dir_name)[-1].split('.')[0])
        new_file_name = f"{dir_name}{last_number + 1}{extension}"
    else:
        new_file_name = f"{dir_name}0{extension}"

    new_file_path = os.path.join(output_dir, new_file_name)
    img.save(new_file_path)

    if remove_file:
        source_file_path = os.path.join(input_path, file)
        os.remove(source_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Classifier and Organizer")

    parser.add_argument('--input-path', '-i', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output-path', '-o', type=str, required=True, help='Output directory for organized images')
    parser.add_argument('--weights-path', '-w', type=str, required=True, help='Path to model weights')
    parser.add_argument('--class-names-path', '-c', type=str, required=True, help='Path to class names file')
    parser.add_argument('--remove-file', '-r', action='store_true', default=False,
                        help='Remove the source files after copying (default: False)')

    args = parser.parse_args()

    image_extensions = ['jpg', 'jpeg', 'png']

    class_names = get_class_names(args.class_names_path)
    model, transform = create_effnetb2_model(num_classes=len(class_names))
    load_weights(args.weights_path, model)

    files = os.listdir(args.input_path)

    for file in files:
        if file.split('.')[-1] in image_extensions:
            img, predicted_prob, predicted_class = get_prediction(args.input_path, file, model, transform, class_names)
            move_file(args.input_path, args.output_path, img, predicted_prob, predicted_class, file, args.remove_file)