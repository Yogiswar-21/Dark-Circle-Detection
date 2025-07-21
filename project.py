import sys
from ultralytics import YOLO
import datetime

# Load the model
model = YOLO('/home/rguktrkvalley/Documents/Mini project/best.pt')  # Ensure the correct path to your model

# Advice Dictionary
advice_dict = {
    'high': {
        'causes': ' Lack of sleep\n Dehydration\n Genetics\n Stress and screen time',
        'remedies': ' Sleep at least 7-8 hours\n Stay hydrated\n Use under-eye creams\n Manage stress levels\nLimit screen exposure'
    },
    'moderate': {
        'causes': ' Irregular sleep patterns\n Mild stress\n Slight dehydration',
        'remedies': ' Fix sleep schedule\n Drink plenty of water\nGentle eye massages\n Use cold compress therapy'
    },
    'low': {
        'causes': ' Long screen time\n Mild tiredness',
        'remedies': ' Take regular screen breaks\n Relax your eyes frequently\n Stay hydrated'
    },
    'no_dark_circle': {
        'causes': ' No dark circles detected! Great job!',
        'remedies': ' Continue healthy sleep and hydration habits\n Protect your skin from sun exposure'
    }
}

# Prediction + Advice Function
def detect_dark_circles(image_path):
    results = model.predict(image_path, conf=0.25)
    # Get the first result (if more than one image is used)
    result = results[0]

    # Get the predicted class (e.g., 'low', 'moderate', 'high', 'no_dark_circle')
    predictions = result.boxes.cls.tolist()
    names = model.names  # Class names corresponding to YOLO model
    if predictions:
        class_id = int(predictions[0])
        class_name = names[class_id]
    else:
        class_name = 'no_dark_circle'

    # Fetch advice from the dictionary based on the prediction
    advice = advice_dict.get(class_name, {'causes': 'Unknown', 'remedies': 'No advice available.'})

    # Prepare the output text
    output_text = f"""
    #  Result: **{class_name.replace('_', ' ').capitalize()}**

    ---

    ##  Possible Causes:
    {advice['causes']}

    ---

    ##  Recommended Remedies:
    {advice['remedies']}

    ---
    """
    # Show the image with annotations
    result.show()

    return output_text

# Main function to handle command line arguments and run detection
if __name__ == "__main__":
    # Check if the user provided an image path
    if len(sys.argv) < 2:
        print("Error: Please provide an image path.")
        sys.exit(1)

    # Get the image path from the arguments
    image_path = sys.argv[1]

    # Run the detection and get result
    output_text = detect_dark_circles(image_path)

    # Print the advice text
    print(output_text)

