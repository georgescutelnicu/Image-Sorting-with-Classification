import os
import shutil
import torch
from sorter import get_class_names, load_weights, get_prediction, move_file
from PIL import Image
from model_builder import create_effnetb2_model


CLASS_NAMES_PATH = "tests/class_names.txt"
WEIGHTS_PATH = "tests/model/model_v4.pth"
DATA_TRAIN_DIR = "tests/data/train"
TESTING_IMAGE = "tests/data/test/snow/snow1.jpg"


def test_get_class_names():
    class_names = [folder_name for folder_name in os.listdir(DATA_TRAIN_DIR)]

    with open(CLASS_NAMES_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    for name in class_names:
        assert name in get_class_names(CLASS_NAMES_PATH)


def test_load_weights():
    loaded_model, _ = create_effnetb2_model(num_classes=12)
    load_weights(WEIGHTS_PATH, loaded_model)

    saved_state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"))

    for key in saved_state_dict:
        assert torch.equal(saved_state_dict[key], loaded_model.state_dict()[key])


def test_get_prediction(model):
    folder, image = "/".join(TESTING_IMAGE.split("/")[:-1]) + "/", TESTING_IMAGE.split("/")[-1]
    class_names = get_class_names(CLASS_NAMES_PATH)
    model, transform = model

    img, predicted_prob, predicted_class = get_prediction(folder, image, model, transform, class_names)

    assert isinstance(img, Image.Image)
    assert isinstance(predicted_prob, float)
    assert predicted_class in class_names


def test_move_file(model):
    folder, image = "/".join(TESTING_IMAGE.split("/")[:-1]) + "/", TESTING_IMAGE.split("/")[-1]
    class_names = get_class_names(CLASS_NAMES_PATH)
    model, transform = model

    img, predicted_prob, predicted_class = get_prediction(folder, image, model, transform, class_names)

    output_path = "tests/temp_folder"
    expected_path = os.path.join(output_path, "unknown" if predicted_prob < 0.35 else predicted_class)
    expected_file = f"{predicted_class}0.jpg" if predicted_prob >= 0.35 else "unknown0.jpg"

    move_file(folder, output_path, img, predicted_prob, predicted_class, image, remove_file=False)

    assert os.path.exists(expected_path)
    assert os.path.exists(os.path.join(expected_path, expected_file))

    os.remove(CLASS_NAMES_PATH)
    shutil.rmtree(output_path)

    assert not os.path.exists(expected_path)
    assert not os.path.exists(os.path.join(expected_path, expected_file))
