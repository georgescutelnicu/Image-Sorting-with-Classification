## Deep Learning Image Classification Project

This comprehensive project combines deep learning image classification with a file organization script, leveraging the power of PyTorch and the EfficientNetB2 architecture for image classification while providing a convenient way to organize your image files.

<img src='extras/nn.png'>

## Project Highlights
- This project is tailored for image classification tasks and offers a complete solution for training and deploying deep learning models effectively.
- It includes a custom implementation of the EfficientNetB2 architecture for image classification, which can be adapted for various datasets.
- The project features a file organization tool that classifies images into relevant folders based on their content.
  
### **The project's model has been trained to recognize multiple weather conditions including:**
- Cloudy
- Rain
- Snow
- Fogg
- Dew
- Tornado
- Frost
- Hail
- Lightning
- Rainbow
- Shine
- Sunrise

### **File Organization**
- Use the script to organize your image files easily. Simply provide the path of the input_dir(Input directory containing images), output_dir(Output directory for organized images), model_weights, and class_names, and the script will classify the images accordingly. To remove the source files after copying, add the `-r` flag.

## Getting Started
- To train the image classification model, please refer to the accompanying Jupyter notebook in this repository.
- To organize your files, follow these steps:
  1. Ensure you have the required libraries installed (e.g., PyTorch).
  2. Run the script with the following command:
     ```
     python main.py --input-path 'path/to/your/input/directory' --output-path 'path/to/your/output/directory' --weights-path 'path/to/your/model/weights' --class-names-path 'path/to/your/class_names.txt' -r
     ```
     - Replace `'path/to/your/input/directory'` with the directory containing your unorganized image files.
     - Replace `'path/to/your/output/directory'` with the directory where you want to organize the images.
     - Replace `'path/to/your/model/weights'` with the path to your model weights.
     - Replace `'path/to/your/class_names.txt'` with the path to your class names file.
     - Add the `-r` flag to remove source files after copying if desired.

## Technologies Used
- **Python**: The core programming language used for implementing the project's modules and scripts.
- **PyTorch**: A powerful deep learning framework used for building and training neural networks.
- **EfficientNetB2**: A state-of-the-art deep learning architecture employed for image classification tasks.
- **Matplotlib**: A versatile library for creating visualizations, including plotting loss and accuracy curves during training.

## Accuracy
<img src='extras/loss_curve.png'>

**This project provides a solid foundation for building and training deep neural networks for image classification tasks and offers a convenient way to organize image files. Feel free to use and modify it for your specific image classification and file organization projects.**

*For more details on model training, please refer to the accompanying Jupyter notebook in this repository.*

### [[DEMO]](https://huggingface.co/spaces/georgescutelnicu/weather-image-classifier)
