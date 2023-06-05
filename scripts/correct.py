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
        self.last_checked_time = time.time()

    def title(self):
        return "Aloe Referencer"

    def show(self, is_txt2img):
        return scripts.AlwaysVisible

    def ui(self, _=None):
        self.directory_input = gr.inputs.Textbox(label='Path to the images directory')
        self.reference_image_input = gr.inputs.Image(label="Upload reference image")
        self.enabled_checkbox = gr.inputs.Checkbox(False, label="Enable/Disable")
        self.run_script_button = gr.Button(label="Run script")
        self.run_script_button.click(self.run)

        return [self.directory_input, self.reference_image_input, self.enabled_checkbox, self.run_script_button]

    def check_new_images(self, directory):
        current_time = time.time()
        if current_time - self.last_checked_time >= 60:  # Check every minute
            self.last_checked_time = current_time
            images = os.listdir(directory)
            images = [image for image in images if image.endswith(('.jpg', '.png', '.jpeg'))]
            if len(images) > 0:
                return images[-1]  # Return the latest image in the directory
        return None

    def run(self, directory, reference_image, enabled):
        self.enabled = enabled
        weight = 0.5  # Default value

        if self.enabled:
            new_image = self.check_new_images(directory)
            if new_image:
                # Convert reference_image to OpenCV format
                reference_img = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
                # Load the new image
                new_image_path = os.path.join(directory, new_image)
                last_image = cv2.imread(new_image_path)
                # Adjust the last image to match the reference image
                adjusted_img = adjust_image_to_reference(last_image, reference_img)
                # Convert back to PIL Image for further enhancements
                adjusted_img_pil = Image.fromarray(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
                # Apply contrast and sharpness with weight
                final_img = apply_contrast_and_sharpness(adjusted_img_pil, weight, weight)
                # Save the final image
                final_image_path = os.path.splitext(new_image_path)[0] + '_edit' + os.path.splitext(new_image_path)[1]
                final_img.save(final_image_path)
                # Convert final image to Gradio interface format
                final_img = Image.open(final_image_path)
                return final_img
            else:
                print("No new images found since the script was enabled.")
        else:
            print("Script is currently disabled.")
            self.last_checked_time = time.time()

        return None
