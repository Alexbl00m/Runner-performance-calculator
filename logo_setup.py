import os
import io
from PIL import Image, ImageDraw, ImageFont

def create_logo():
    """
    Creates a branded logo for the Runner Performance Calculator
    Returns the path to the logo file.
    """
    logo_path = "Logotype_Light@2x.png"
    
    # Check if logo already exists
    if os.path.exists(logo_path):
        return logo_path
    
    # If not, create a branded logo
    try:
        # Create a transparent background
        img = Image.new('RGBA', (400, 200), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font if available, otherwise use default
        try:
            # Try common font names - adjust as needed
            font_title = ImageFont.truetype("Arial Bold.ttf", 50)
            font_subtitle = ImageFont.truetype("Arial.ttf", 30)
        except IOError:
            try:
                # Try alternative font names
                font_title = ImageFont.truetype("arialbd.ttf", 50)
                font_subtitle = ImageFont.truetype("arial.ttf", 30)
            except IOError:
                # Fall back to default
                font_title = ImageFont.load_default()
                font_subtitle = ImageFont.load_default()
        
        # Brand color (orange from your styling)
        brand_color = (230, 117, 78)  # E6754E in RGB
        secondary_color = (44, 62, 80)  # 2C3E50 in RGB
        
        # Draw company name
        draw.text((20, 50), "RUNNER", fill=brand_color, font=font_title)
        draw.text((20, 110), "CALCULATOR", fill=secondary_color, font=font_subtitle)
        
        # Add a decorative element
        draw.line([(20, 100), (380, 100)], fill=brand_color, width=3)
        
        # Add a runner icon (simple silhouette)
        runner_points = [
            (320, 60),  # head
            (310, 75),  # shoulder
            (325, 80),  # arm back
            (300, 95),  # body
            (290, 115),  # leg forward
            (310, 130),  # foot forward
            (315, 100),  # leg back
            (330, 120),  # foot back
        ]
        
        # Draw runner
        for i in range(len(runner_points) - 1):
            draw.line([runner_points[i], runner_points[i+1]], fill=brand_color, width=3)
        
        # Close the shape
        draw.line([runner_points[-1], runner_points[0]], fill=brand_color, width=3)
        
        # Save the image
        img.save(logo_path)
        return logo_path
    
    except Exception as e:
        print(f"Error creating logo: {e}")
        return None

if __name__ == "__main__":
    create_logo()
