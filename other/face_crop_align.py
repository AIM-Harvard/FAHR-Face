import os
from PIL import Image
from retinaface import RetinaFace

# Define the directories
input_dir = "/path/to/input/folder/"
output_dir = "/path/to/output/folder/"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of filenames
filenames = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]  # Adjust as needed

# Loop over the images in the input directory
for filename in filenames:
    image_path = os.path.join(input_dir, filename)
    resp = RetinaFace.detect_faces(image_path)

    # Check if exactly one face is detected
    if resp is not None and len(resp) == 1:
        # Extracting and aligning the face
        faces = RetinaFace.extract_faces(img_path=image_path, align=True)

        for face in faces:
            # Convert face (numpy array) to PIL Image
            face_image = Image.fromarray(face)

            # Save the cropped and aligned face image in the highest quality
            output_path = os.path.join(output_dir, filename)
            face_image.save(output_path, quality=95)  # Adjust the quality parameter as needed (0-95)
            break  # Only save the first face (since there should only be one)
    else:
        print(f"Skipping {filename} as no face or multiple faces were detected.")

print("Preprocessing completed.")