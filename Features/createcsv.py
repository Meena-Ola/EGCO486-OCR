import os
import csv
def create_dataset_csv(image_dir, output_csv):
    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Open the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'label'])  # Write the header

        for image_file in image_files:
            # Extract the label from the image file name (assuming the label is part of the file name)
            label = os.path.splitext(image_file)[0]  # Remove the file extension to get the label
            image_path = os.path.join(image_dir, image_file)
            writer.writerow([image_path, label])

    print(f"CSV file saved as {output_csv}")
