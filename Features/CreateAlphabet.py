from PIL import Image, ImageDraw, ImageFont
import os

# Function to create alphabet images
def create_alphabet_images(output_dir='alphabet_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the alphabet (upper and lower case)
    alphabets = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

    # Create a simple image for each character
    for char in alphabets:
        # Create a blank white image (size 64x64, white background)
        img = Image.new('RGB', (64, 64), color='white')
        
        # Initialize ImageDraw object
        draw = ImageDraw.Draw(img)
        
        # Use a built-in font (you can change this to any other font file)
        try:
            font = ImageFont.truetype("arial.ttf", 40)  # Adjust font size as needed
        except IOError:
            font = ImageFont.load_default()
        
        # Get the size of the text to center it
        text_width, text_height = img.size
        position = ((64 - text_width) // 2, (64 - text_height) // 2)
        
        # Draw the character in black
        draw.text(position, char, fill='black', font=font)
        
        # Save the image
        img.save(f'{output_dir}/{char}.png')

# Create sample alphabet images
create_alphabet_images(output_dir="SampleIMG")
