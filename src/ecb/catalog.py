"""Rule-based product categories from free-text descriptions.

Online Retail II has no category column, so categories are derived from description keywords.
Rules are checked in order; the first match wins (e.g. 'CHRISTMAS CANDLE' is seasonal, not lighting).
"""
import re

RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Christmas & Seasonal", ("CHRISTMAS", "XMAS", "ADVENT", "EASTER", "SNOWFLAKE", "REINDEER",
                              "SANTA", "HALLOWEEN", "BAUBLE", "NATIVITY")),
    ("Toys & Kids", ("TOY", "TOYS", "GAME", "DOLL", "DOLLY", "SPACEBOY", "KIDS", "CHILDRENS",
                     "CHILDREN'S", "JIGSAW", "PUZZLE", "SKIPPING", "CRAYONS", "FELTCRAFT", "TEDDY")),
    ("Party & Gift Wrap", ("PARTY", "BUNTING", "BIRTHDAY", "WRAP", "BALLOON", "BALLOONS", "GIFT",
                           "CONFETTI", "RIBBON", "RIBBONS")),
    ("Stationery & Cards", ("CARD", "CARDS", "NOTEBOOK", "PEN", "PENS", "PENCIL", "PENCILS", "PAPER",
                            "STICKER", "STICKERS", "ERASER", "CALENDAR", "BOOK", "ENVELOPE", "STAMP")),
    ("Candles & Lighting", ("CANDLE", "CANDLES", "T-LIGHT", "T-LIGHTS", "TEALIGHT", "LIGHT", "LIGHTS",
                            "LANTERN", "LAMP", "NIGHTLIGHT", "CANDLEHOLDER")),
    ("Home Textiles", ("CUSHION", "THROW", "APRON", "TOWEL", "BLANKET", "QUILT", "HOT WATER BOTTLE",
                       "TEA TOWEL", "OVEN GLOVE")),
    ("Kitchen & Dining", ("MUG", "MUGS", "CUP", "CUPS", "TEA", "TEACUP", "PLATE", "PLATES", "BOWL",
                          "BOWLS", "CAKE", "CAKES", "BAKING", "CUTLERY", "NAPKIN", "NAPKINS", "SPOON",
                          "SPOONS", "JUG", "TEAPOT", "LUNCH", "BOTTLE", "GLASS", "TUMBLER", "CAKESTAND",
                          "CAKESTAND", "TRAY", "COASTER", "COASTERS", "KITCHEN", "PANTRY", "JAM", "EGG")),
    ("Jewellery & Accessories", ("NECKLACE", "NECKLAC", "EARRINGS", "BRACELET", "RING", "HAIR",
                                 "CHARM", "PURSE", "UMBRELLA", "BROOCH", "KEYRING", "KEY-RING")),
    ("Garden & Outdoor", ("GARDEN", "PLANT", "PLANTER", "BIRD", "BIRDHOUSE", "WATERING", "DOORMAT",
                          "POT", "FLOWER", "PICNIC")),
    ("Bags & Storage", ("BAG", "BAGS", "BASKET", "STORAGE", "BOX", "BOXES", "TIN", "TINS", "JAR",
                        "HOLDER", "DRAWER", "CASE", "CASES", "CRATE", "HAMPER")),
]
DEFAULT = "Home Decor"

_PATTERNS = [(cat, re.compile(r"(?<![A-Z0-9'-])(" + "|".join(map(re.escape, kws)) + r")(?![A-Z0-9'-])"))
             for cat, kws in RULES]


def categorise(description: str | None) -> str:
    text = (description or "").upper()
    for cat, pat in _PATTERNS:
        if pat.search(text):
            return cat
    return DEFAULT


# Synthetic gross-margin assumptions per category (the source has no cost data).
MARGIN = {
    "Christmas & Seasonal": 0.55, "Toys & Kids": 0.45, "Party & Gift Wrap": 0.60,
    "Stationery & Cards": 0.62, "Candles & Lighting": 0.50, "Home Textiles": 0.48,
    "Kitchen & Dining": 0.42, "Jewellery & Accessories": 0.65, "Garden & Outdoor": 0.40,
    "Bags & Storage": 0.47, "Home Decor": 0.52,
}
