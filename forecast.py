# =========================
# MONTHLY RAIN ADJUSTMENT
# =========================

def adjust_monthly_rain(markov_forecast, adjustment):

    levels = [
        "No Rain",
        "Light Rain",
        "Moderate Rain",
        "Heavy Rain"
    ]

    if markov_forecast not in levels:
        return markov_forecast

    idx = levels.index(markov_forecast)

    if adjustment <= -4:
        shift = -2
    elif adjustment <= -2:
        shift = -1
    elif adjustment >= 4:
        shift = 2
    elif adjustment >= 2:
        shift = 1
    else:
        shift = 0

    idx += shift
    idx = max(0, min(idx, len(levels) - 1))

    return levels[idx]
def translate_rain(state, language):

    translations = {
        "No Rain": "Hakuna Mvua",
        "Light Rain": "Mvua Ndogo",
        "Moderate Rain": "Mvua za Wastani",
        "Heavy Rain": "Mvua Kubwa"
    }

    if language == "Kiswahili":
        return translations.get(state, state)

    return state


def translate_faw(state, language):

    translations = {
        "High": "Kubwa",
        "Medium": "Wastani",
        "Low": "Ndogo"
    }

    if language == "Kiswahili":
        return translations.get(state, state)

    return state


def translate_vpd(state, language):

    translations = {
        "Dry": "Kavu",
        "Good": "Nzuri",
        "Humid": "Unyevunyevu"
    }

    if language == "Kiswahili":
        return translations.get(state, state)

    return state
