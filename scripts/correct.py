import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageStat
import time

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.scripts as scripts

class AloeReferencer(scripts.Script):
    def __init__(self):
        self.enabled = False

    def title(self):
        return "Aloe Referencer"

    def show(self, is_txt2img):
        return scripts.AlwaysVisible

    def ui(self, _=None):
        self.directory_input = gr.inputs.Textbox(label='Path to the images directory')
        self.reference_image_input = gr.inputs.Image(label="Upload reference image")
        self.enabled_checkbox = gr.inputs.Checkbox(False, label="Enable/Disable")
        self.enabled_checkbox.select(self.update_enabled_status)
        self.run_script_button = gr.Button(label="Run script")
        self.run_script_button.click(self.run)

        return [self.directory_input, self.reference_image_input, self.enabled_checkbox, self.run_script_button]

    def update_enabled_status(self):
        self.enabled = not self.enabled
        return self.enabled

    def get_last_image(self, directory):
        images = os.listdir(directory)
        images = [image for image in images if image.endswith(('.jpg', '.png', '.jpeg'))]
        if len(images) > 0:
            return os.path.join(directory, images[-1])  # Return the latest image in the directory
        return None

    def run(self, directory, reference_image):
        weight = 0.5  # Default value

        if self.enabled:
            # Load reference image
            reference_img = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)

            # Get the last image in the directory
            last_image_path = self.get_last_image(directory)
            if not last_image_path:
                print('No images found in the directory.')
                return

            # Load the last image
            last_image = cv2.imread(last_image_path)

            # Adjust the last image to match the reference image
            adjusted_img = adjust_image_to_reference(last_image, reference_img)
            # Convert back to PIL Image for further enhancements
            adjusted_img_pil = Image.fromarray(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
            # Apply contrast and sharpness with weight
            final_img = apply_contrast_and_sharpness(adjusted_img_pil, weight, weight)
            # Save the final image
            final_image_path = os.path.splitext(last_image_path)[0] + '_edit' + os.path.splitext(last_image_path)[1]
            final_img.save(final_image_path)
            # Convert final image to Gradio interface format
            final_img = Image.open(final_image_path)
            return final_img
        else:
            print("Script is currently disabled.")
        return None
