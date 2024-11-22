import pandas as pd

# Define the mapping dictionary
char_to_int = {idx:char  for idx, char in enumerate(' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')}
# Load the CSV file
csv_file_path = 'E:\CODE\Image Processing\Image-Processing-Project-\English\Fnt\english.csv'
data = pd.read_csv(csv_file_path)

# Apply the mapping to the 'label' column
data['label'] = data['label'].apply(lambda x: char_to_int[x])

# Save the updated CSV file
updated_csv_file_path = 'english.csv'
data.to_csv(csv_file_path, index=False)

print(f'Updated CSV file saved to {updated_csv_file_path}')