import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def rainfall_state(rain):

    if rain < 1:
        return "No Rain"
    elif rain < 20:
        return "Light Rain"
    elif rain < 50:
        return "Moderate Rain"
    else:
        return "Heavy Rain"


def ai_image(state_type, state):

    image_map = {

        "FAW": {
            "High": "faw_high.jpg",
            "Medium": "faw_medium.jpg",
            "Low": "faw_low.jpg"
        },

        "VPD": {
            "Dry": "dry.jpg",
            "Good": "good.jpg",
            "Humid": "humid.jpg"
        },

        "RAIN": {
            "No Rain": "no_rain.jpg",
            "Light Rain": "rain_light.jpg",
            "Moderate Rain": "rain_moderate.jpg",
            "Heavy Rain": "rain_heavy.jpg"
        }

    }

    filename = image_map[state_type][state]

    return os.path.join(
        BASE_DIR,
        "images",
        filename
    )
    
