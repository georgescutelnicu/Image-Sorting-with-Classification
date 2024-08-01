from model_builder import create_effnetb2_model
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
import os
import argparse


def get_class_names(class_names_path):
    """
        Load class names from a file.

        Parameters:
            class_names_path (str): Path to the file containing class names.

        Returns:
            list: List of class names.
    """

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    return class_names


def load_weights(weights_path, model):
    """
        Load pre-trained weights into a model.

        Parameters:
            weights_path (str): Path to the model weights file.
            model (torch.nn.Module): The model to load the weights into.
    """

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()


def get_prediction(input_path, file, model, transform, class_names):
    """
        Get predictions for an input image.

        Parameters:
            input_path (str): Path to the input directory.
            file (str): Filename of the image.
            model (torch.nn.Module): The pre-trained image classification model.
            transform (torchvision.transforms.Compose): Image transformation pipeline.
            class_names (list): List of class names.

        Returns:
            tuple: Tuple containing the image, predicted probability, and predicted class.
    """

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
    """
        Move an image file to the appropriate directory based on the predictions.

        Parameters:
            input_path (str): Path to the input directory.
            output_path (str): Path to the output directory.
            img (PIL.Image.Image): The image to be moved.
            predicted_prob (float): Predicted probability of the image.
            predicted_class (str): Predicted class of the image.
            file (str): Filename of the image.
            remove_file (bool): Whether to remove the source file after copying (default: False).
    """

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
