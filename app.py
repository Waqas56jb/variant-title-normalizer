import pandas as pd
import re
import nltk
from fuzzywuzzy import fuzz
import os
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from spellchecker import SpellChecker
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK data for tokenization and POS tagging
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize spell checker
spell_checker = SpellChecker(language='en')

# Enhanced regex patterns for attribute extraction
REGEX_PATTERNS = {
    "Size": r"\b(size|s-m-l|xs|s|m|l|xl|xxl|xxxl|xxxxl|2xs-4xl|inch|in|cm|mm|ft|oz|lb|kg|g|diameter|length|height|width|depth|back length|bead height|hoop diameter|strap size|neck size|waist size|drop length|cuff size|ring size|undies size|hxw|h x w|l x w x h|measured in thickness|taille|grandeur|talla|größe|misura|fan size|belt size|collar size|inseam|chest size|shoe size|sleeve length|small|medium|large|petite|plus|junior|one size|custom size|\[in\]|\[cm\]|\[ft\]|\[mm\])\b",
    "Color": r"\b(color|colour|shade|hue|black|white|red|blue|green|yellow|pink|grey|gray|olive|cream|mulberry|silver|gold|bronze|copper|platinum|navy|teal|lavender|maroon|beige|ivory|burgundy|indigo|violet|cyan|magenta|rose|sand|rust|amber|emerald|ruby|sapphire|multicolor|camouflage|polka dot|plaid|checkered|floral|geometric|tie-dye|gradient|ombre|colorblock|stripes|metal color|frame color|buckle color|trim color|stitching color|pattern color|flat bar metal)\b",
    "Material": r"\b(material|fabric|cotton|leather|silk|wool|polyester|nylon|spandex|denim|linen|velvet|satin|chiffon|lace|suede|canvas|corduroy|jersey|knit|cashmere|viscose|rayon|bamboo|rubber|vinyl|pu leather|faux leather|stainless steel|titanium|aluminum|copper|brass|wood|glass|ceramic|porcelain|marble|stone|resin|plastic|silicone|latex|neoprene|fill|cover|shell|stuffing|goose down|polyfill|memory foam|cotton fill|wool fill|faux fur|real fur|pearl|gemstone|crystal|rhinestone|metal|alloy|enamel|varnish|lacquer|upholstery|padding|lining|weave|knit stitch|embroidery|appliqué|quilted|waterproof|breathable|sari|embroidered|charm|flat bar)\b",
    "Style": r"\b(style|fit type|slim fit|regular fit|relaxed fit|skinny fit|bootcut|flare|wide leg|straight leg|cropped|midi|maxi|mini|off-shoulder|one-shoulder|v-neck|crew neck|high neck|turtleneck|collar|reversible|hooded|wrap|peplum|asymmetric hem|double-breasted|single-breasted|short sleeve|long sleeve|cap sleeve|bell sleeve|drape|pleated|ruched|smocked|belted|drawstring|zipper fly|button fly|buckle|clasp|pattern|floral|geometric|abstract|camouflage|polka dot|plaid|checkered|stripes|sienna|bohemian|casual|formal|elegant|boho|sporty|vintage|modern|classic|hoodie|jogger|pant|shirt|skirt|top|single pearl|five pearl|charm|design|diy floor kit)\b",
    "Finish": r"\b(finish|polish|matte|glossy|satin finish|brushed finish|polished finish|hammered finish|textured finish|distressed finish|antique finish|vintage finish|patinated finish|anodized finish|powder coated|electroplated|engraved finish|lacquered finish|varnished finish|painted finish|natural finish|jewelry finish|color & finish)\b",
    "Pack Size": r"\b(pack size|quantity|count|units per pack|pack type|single|pair|double|triple|quad|set|bundle|multipack|value pack|trial pack|bulk pack|pack of|set of|combo pack|assorted pack|gift set|bundle size|arne collection)\b",
    "Hardware Type": r"\b(hardware|clasp type|closure type|buckle type|zipper type|snap type|hook type|ring type|chain type|cord type|strap type|hinge type|lock type|latch type|fastener type|screw type|bolt type|nut type|clip type)\b",
    "Scent": r"\b(scent|fragrance|aroma|floral|fruity|citrus|woody|spicy|herbal|fresh|sweet|musky|amber|vanilla|lavender|rose|jasmine|cedarwood|sandalwood|bergamot|lemon|orange|apple|peach|coconut|pineapple|cinnamon|clove|mint|eucalyptus|coffee|chocolate|caramel|honey|clean linen|fresh air|ocean breeze)\b",
    "Flavor": r"\b(flavor|taste|original|classic|vanilla|chocolate|strawberry|raspberry|blueberry|lemon|lime|orange|mint|peppermint|caramel|coffee|mocha|cinnamon|pumpkin spice|apple pie|cherry|grape|watermelon|mango|single origins)\b",
    "Voltage": r"\b(voltage|power|v|wattage|watts|amps|ac|dc|dual voltage|110v|120v|220v|240v|100-240v|50hz|60hz|12v|24v|5v|9v|low voltage|high voltage|corded|cordless|battery powered|rechargeable|aa|aaa|cr2032|usb charger)\b",
    "Connectivity": r"\b(connectivity|plug type|cable length|usb|bluetooth|wifi|wireless|wired|ethernet|hdmi|displayport|thunderbolt|usb-a|usb-c|micro-usb|bluetooth 5.0|wifi 6|2.4ghz|5ghz|dual band|remote controlled|voice controlled|app controlled)\b",
    "Compatibility": r"\b(compatibility|model|year|ios|android|windows|mac|universal|specific model|version|generation|series|edition)\b",
    "Theme": r"\b(theme|edition|motif|seasonal|holiday|christmas|halloween|easter|valentines|wedding|party|travel|adventure|nautical|boho|vintage|retro|modern|minimalist|sin print|tic-tac-toe|paw|shoe|wine glass)\b",
    "Age Group": r"\b(age group|age|kids|children|teens|adult|senior|infant|toddler|youth|baby|newborn|junior|senior citizen|all ages)\b",
    "Strength": r"\b(strength|level|potency|firmness|soft|medium|hard|extra firm|low|medium strength|high strength|mild|strong|intense|lightweight|heavy duty)\b",
    "Variety": r"\b(variety|type|variant|options|assorted|mixed|standard|premium|deluxe|basic|advanced|pro|lite|mini|full size|travel size|custom)\b",
    "Weight Capacity": r"\b(weight capacity|weight|load|max weight|capacity|lb|kg|oz|g|ton|light duty|medium duty|heavy duty)\b",
    "Language": r"\b(language|letra|letter|english|spanish|french|german|italian|multilingual)\b",
    "Letter": r"\b(letter|letra|a-z|A-Z|initial|monogram)\b",
    "Formula Type": r"\b(formula|skin type|cream|gel|lotion|oil|serum|powder|liquid|spray|foam|capsule|tablet|patch|organic|vegan|natural|synthetic|hypoallergenic)\b",
    "Target Use": r"\b(target use|usage type|application|daily|night|morning|indoor|outdoor|travel|home|office|gym|kitchen|bathroom|bedroom|professional|casual|formal)\b",
    "Caffeine Level": r"\b(caffeine level|caffeinated|decaf|low caffeine|high caffeine|caffeine free)\b",
    "Tip Type": r"\b(tip type|brush|pen|fine tip|medium tip|broad tip|chisel tip|bullet tip|felt tip|rollerball|gel pen|ballpoint)\b",
    "Delivery Type": r"\b(delivery type|spray|capsule|tablet|liquid|powder|gel|patch|inhaler|dropper|syringe|pump|roll-on|stick)\b",
    "Storage Size": r"\b(storage size|capacity|gb|tb|mb|kb|byte|16gb|32gb|64gb|128gb|256gb|512gb|1tb|2tb|small capacity|large capacity)\b",
    "Model Year": r"\b(model year|year|edition|2020|2021|2022|2023|2024|2025|new model|old model|latest model)\b",
    "Fit Type": r"\b(fit type|slim fit|regular fit|relaxed fit|loose fit|skinny fit|athletic fit|tailored fit|stretch fit|non-stretch)\b",
    "Sleeve Length": r"\b(sleeve length|short sleeve|long sleeve|three-quarter sleeve|cap sleeve|sleeveless|full sleeve)\b",
    "Clasp Type": r"\b(clasp type|closure type|lobster clasp|spring ring|toggle clasp|box clasp|hook clasp|magnetic clasp|snap clasp)\b",
    "Jewelry Finish": r"\b(jewelry finish|polish|matte finish|glossy finish|satin finish|brushed finish|hammered finish)\b",
    "Pattern": r"\b(pattern|sin print|stripes|colorblock|floral|geometric|abstract|polka dot|plaid|checkered|herringbone|houndstooth|chevron|zigzag)\b",
    "Shade": r"\b(shade|hue|light|dark|medium|pastel|neon|muted|bright|vibrant|soft)\b",
    "Skin Type": r"\b(skin type|dry|oily|sensitive|normal|combination|acne-prone|aging|mature)\b",
    "Plug Type": r"\b(plug type|usb|type-c|micro-usb|lightning|universal plug|eu plug|us plug|uk plug|au plug)\b",
    "Cable Length": r"\b(cable length|ft|m|cm|inch|1m|2m|3ft|6ft|short cable|long cable|extendable)\b",
    "Quantity": r"\b(quantity|count|single|pair|double|triple|quad|set|pack|bundle|1 piece|2 pieces|3 pieces|multiple)\b",
    "Closure Type": r"\b(closure type|clasp type|buckle type|zipper|snap|hook-and-eye|button|velcro|drawstring|elastic)\b",
    "Edition": r"\b(edition|version|limited edition|special edition|collector’s edition|standard edition|deluxe edition)\b",
    "Collection": r"\b(collection|arne collection|series|seasonal collection|capsule collection|signature collection)\b",
    "Composition": r"\b(composition|makeup|build|blend|mixture|formula|100%|pure|mixed|synthetic blend|natural blend)\b",
}

# Taxonomy with hierarchical structure
TAXONOMY = {
    "Size": {
        "Clothing": ["xs", "s", "m", "l", "xl", "xxl", "xxxl", "xxxxl", "2xs-4xl", "petite", "plus", "junior", "one size", "s-m-l"],
        "Measurements": ["inch", "cm", "mm", "ft", "oz", "lb", "kg", "g", "diameter", "length", "height", "width", "depth", "back length", "bead height", "hoop diameter"],
        "Accessories": ["strap size", "neck size", "waist size", "drop length", "cuff size", "ring size", "undies size", "fan size", "belt size", "collar size", "inseam", "chest size", "shoe size", "sleeve length"],
        "General": ["small", "medium", "large", "custom size", "taille", "grandeur", "talla", "größe", "misura"]
    },
    "Color": {
        "Basic": ["black", "white", "red", "blue", "green", "yellow", "pink", "grey", "gray", "olive", "cream", "mulberry", "charcoal", "neon"],
        "Metallic": ["silver", "gold", "bronze", "copper", "platinum"],
        "Shades": ["navy", "teal", "lavender", "maroon", "beige", "ivory", "burgundy", "indigo", "violet", "cyan", "magenta", "rose", "sand", "rust", "amber", "emerald", "ruby", "sapphire"],
        "Patterns": ["multicolor", "camouflage", "polka dot", "plaid", "checkered", "floral", "geometric", "tie-dye", "gradient", "ombre", "colorblock", "stripes"],
        "Special": ["metal color", "frame color", "buckle color", "trim color", "stitching color", "pattern color", "flat bar metal"]
    },
    "Material": {
        "Fabric": ["cotton", "leather", "silk", "wool", "polyester", "nylon", "spandex", "denim", "linen", "velvet", "satin", "chiffon", "lace", "suede", "canvas", "corduroy", "jersey", "knit", "cashmere", "viscose", "rayon", "bamboo", "sari", "embroidered"],
        "Synthetic": ["rubber", "vinyl", "pu leather", "faux leather", "plastic", "silicone", "latex", "neoprene"],
        "Metal": ["stainless steel", "titanium", "aluminum", "copper", "brass", "metal", "alloy", "flat bar"],
        "Natural": ["wood", "glass", "ceramic", "porcelain", "marble", "stone", "resin"],
        "Fill": ["fill", "cover", "shell", "stuffing", "goose down", "polyfill", "memory foam", "cotton fill", "wool fill"],
        "Special": ["faux fur", "real fur", "pearl", "gemstone", "crystal", "rhinestone", "enamel", "varnish", "lacquer", "upholstery", "padding", "lining", "weave", "knit stitch", "embroidery", "appliqué", "quilted", "waterproof", "breathable", "charm"]
    },
    "Style": {
        "Fit": ["slim fit", "regular fit", "relaxed fit", "skinny fit", "bootcut", "flare", "wide leg", "straight leg", "cropped", "midi", "maxi", "mini"],
        "Neckline": ["off-shoulder", "one-shoulder", "v-neck", "crew neck", "high neck", "turtleneck", "collar"],
        "Sleeve": ["short sleeve", "long sleeve", "three-quarter sleeve", "cap sleeve", "sleeveless", "bell sleeve"],
        "Design": ["reversible", "hooded", "wrap", "peplum", "asymmetric hem", "double-breasted", "single-breasted", "drape", "pleated", "ruched", "smocked", "belted", "drawstring", "zipper fly", "button fly", "single pearl", "five pearl", "charm", "design", "diy floor kit"],
        "Pattern": ["floral", "geometric", "abstract", "camouflage", "polka dot", "plaid", "checkered", "stripes", "sienna"],
        "Aesthetic": ["bohemian", "casual", "formal", "elegant", "boho", "sporty", "vintage", "modern", "classic"],
        "Apparel Type": ["hoodie", "jogger", "pant", "shirt", "skirt", "top"]
    },
    "Finish": {
        "Surface": ["matte", "glossy", "satin finish", "brushed finish", "polished finish", "hammered finish", "textured finish", "distressed finish"],
        "Special": ["antique finish", "vintage finish", "patinated finish", "anodized finish", "powder coated", "electroplated", "engraved finish", "lacquered finish", "varnished finish", "painted finish", "natural finish", "jewelry finish", "color & finish"]
    },
    "Pack Size": {
        "Units": ["quantity", "count", "single", "pair", "double", "triple", "quad", "set", "bundle", "multipack", "value pack", "trial pack", "bulk pack", "pack of", "set of", "combo pack", "assorted pack", "gift set"],
        "Type": ["pack size", "pack type", "bundle size", "arne collection"]
    },
    "Hardware Type": {
        "Closure": ["clasp type", "closure type", "buckle type", "zipper type", "snap type", "hook type"],
        "Attachment": ["ring type", "chain type", "cord type", "strap type", "hinge type", "lock type", "latch type", "fastener type"],
        "Fixing": ["screw type", "bolt type", "nut type", "clip type"]
    },
    "Scent": {
        "Type": ["floral", "fruity", "citrus", "woody", "spicy", "herbal", "fresh", "sweet", "musky"],
        "Specific": ["amber", "vanilla", "lavender", "rose", "jasmine", "cedarwood", "sandalwood", "bergamot", "lemon", "orange", "apple", "peach", "coconut", "pineapple", "cinnamon", "clove", "mint", "eucalyptus", "coffee", "chocolate", "caramel", "honey"],
        "General": ["clean linen", "fresh air", "ocean breeze", "scent", "fragrance", "aroma"]
    },
    "Flavor": {
        "Basic": ["original", "classic", "vanilla", "chocolate", "strawberry", "raspberry", "blueberry", "lemon", "lime", "orange", "mint", "peppermint"],
        "Special": ["caramel", "coffee", "mocha", "cinnamon", "pumpkin spice", "apple pie", "cherry", "grape", "watermelon", "mango", "single origins"],
        "General": ["flavor", "taste"]
    },
    "Voltage": {
        "Power": ["v", "wattage", "watts", "amps", "ac", "dc", "dual voltage", "110v", "120v", "220v", "240v", "100-240v", "50hz", "60hz", "12v", "24v", "5v", "9v"],
        "Type": ["low voltage", "high voltage", "corded", "cordless", "battery powered", "rechargeable", "aa", "aaa", "cr2032", "usb charger"]
    },
    "Connectivity": {
        "Connection": ["usb", "bluetooth", "wifi", "wireless", "wired", "ethernet", "hdmi", "displayport", "thunderbolt", "usb-a", "usb-c", "micro-usb"],
        "Specs": ["bluetooth 5.0", "wifi 6", "2.4ghz", "5ghz", "dual band"],
        "Control": ["remote controlled", "voice controlled", "app controlled"],
        "Cable": ["cable length", "plug type"]
    },
    "Compatibility": {
        "Device": ["ios", "android", "windows", "mac", "universal"],
        "Model": ["specific model", "version", "generation", "series", "edition", "model", "year"]
    },
    "Theme": {
        "Seasonal": ["seasonal", "holiday", "christmas", "halloween", "easter", "valentines"],
        "Event": ["wedding", "party", "travel", "adventure"],
        "Design": ["nautical", "boho", "vintage", "retro", "modern", "minimalist", "sin print", "tic-tac-toe", "paw", "shoe", "wine glass"]
    },
    "Age Group": {
        "General": ["kids", "children", "teens", "adult", "senior", "all ages"],
        "Specific": ["infant", "toddler", "youth", "baby", "newborn", "junior", "senior citizen"]
    },
    "Strength": {
        "Level": ["soft", "medium", "hard", "extra firm", "low", "medium strength", "high strength", "mild", "strong", "intense"],
        "Type": ["lightweight", "heavy duty", "strength", "level", "potency", "firmness"]
    },
    "Variety": {
        "Type": ["assorted", "mixed", "standard", "premium", "deluxe", "basic", "advanced", "pro", "lite"],
        "Size": ["mini", "full size", "travel size", "custom"]
    },
    "Weight Capacity": {
        "Units": ["lb", "kg", "oz", "g", "ton"],
        "Type": ["light duty", "medium duty", "heavy duty", "weight capacity", "weight", "load", "max weight", "capacity"]
    },
    "Language": {
        "General": ["english", "spanish", "french", "german", "italian", "multilingual"],
        "Specific": ["language", "letra", "letter"]
    },
    "Letter": {
        "Type": ["a-z", "A-Z", "initial", "monogram", "letter", "letra"]
    },
    "Formula Type": {
        "Form": ["cream", "gel", "lotion", "oil", "serum", "powder", "liquid", "spray", "foam", "capsule", "tablet", "patch"],
        "Attributes": ["organic", "vegan", "natural", "synthetic", "hypoallergenic"]
    },
    "Target Use": {
        "Time": ["daily", "night", "morning"],
        "Location": ["indoor", "outdoor", "travel", "home", "office", "gym", "kitchen", "bathroom", "bedroom"],
        "Purpose": ["professional", "casual", "formal"]
    },
    "Caffeine Level": {
        "Level": ["caffeinated", "decaf", "low caffeine", "high caffeine", "caffeine free"]
    },
    "Tip Type": {
        "Type": ["brush", "pen", "fine tip", "medium tip", "broad tip", "chisel tip", "bullet tip", "felt tip", "rollerball", "gel pen", "ballpoint"]
    },
    "Delivery Type": {
        "Method": ["spray", "capsule", "tablet", "liquid", "powder", "gel", "patch", "inhaler", "dropper", "syringe", "pump", "roll-on", "stick"]
    },
    "Storage Size": {
        "Units": ["gb", "tb", "mb", "kb", "byte", "16gb", "32gb", "64gb", "128gb", "256gb", "512gb", "1tb", "2tb"],
        "Type": ["small capacity", "large capacity"]
    },
    "Model Year": {
        "Year": ["2020", "2021", "2022", "2023", "2024", "2025"],
        "Type": ["new model", "old model", "latest model", "model year", "year", "edition"]
    },
    "Fit Type": {
        "Type": ["slim fit", "regular fit", "relaxed fit", "loose fit", "skinny fit", "athletic fit", "tailored fit", "stretch fit", "non-stretch"]
    },
    "Sleeve Length": {
        "Length": ["short sleeve", "long sleeve", "three-quarter sleeve", "cap sleeve", "sleeveless", "full sleeve"]
    },
    "Clasp Type": {
        "Type": ["lobster clasp", "spring ring", "toggle clasp", "box clasp", "hook clasp", "magnetic clasp", "snap clasp", "clasp type", "closure type"]
    },
    "Jewelry Finish": {
        "Finish": ["polish", "matte finish", "glossy finish", "satin finish", "brushed finish", "hammered finish", "jewelry finish"]
    },
    "Pattern": {
        "Design": ["stripes", "colorblock", "floral", "geometric", "abstract", "polka dot", "plaid", "checkered", "herringbone", "houndstooth", "chevron", "zigzag", "sin print"]
    },
    "Shade": {
        "Tone": ["light", "dark", "medium", "pastel", "neon", "muted", "bright", "vibrant", "soft", "shade", "hue"]
    },
    "Skin Type": {
        "Type": ["dry", "oily", "sensitive", "normal", "combination", "acne-prone", "aging", "mature"]
    },
    "Plug Type": {
        "Type": ["usb", "type-c", "micro-usb", "lightning", "universal plug", "eu plug", "us plug", "uk plug", "au plug"]
    },
    "Cable Length": {
        "Units": ["ft", "m", "cm", "inch", "1m", "2m", "3ft", "6ft"],
        "Type": ["short cable", "long cable", "extendable"]
    },
    "Quantity": {
        "Units": ["single", "pair", "double", "triple", "quad", "set", "pack", "bundle", "1 piece", "2 pieces", "3 pieces", "multiple"]
    },
    "Closure Type": {
        "Type": ["zipper", "snap", "hook-and-eye", "button", "velcro", "drawstring", "elastic", "clasp type", "buckle type", "closure type"]
    },
    "Edition": {
        "Type": ["limited edition", "special edition", "collector’s edition", "standard edition", "deluxe edition", "edition", "version"]
    },
    "Collection": {
        "Type": ["arne collection", "series", "seasonal collection", "capsule collection", "signature collection"]
    },
    "Composition": {
        "Type": ["100%", "pure", "mixed", "synthetic blend", "natural blend", "composition", "makeup", "build", "blend", "mixture", "formula"]
    }
}

def preprocess_title(title: str) -> str:
    """
    Preprocess the title by trimming whitespace, normalizing dashes, applying title case, removing verbs and prepositions,
    and correcting spelling.

    Args:
        title (str): The raw variant title.

    Returns:
        str: The preprocessed title.
    """
    if not isinstance(title, str) or not title.strip():
        return ""
    title = re.sub(r'\s+', ' ', title.strip())
    title = re.sub(r'[–—]', '-', title)
    title = re.sub(r'\[(.*?)\]', r'\1', title)

    # Tokenize and tag parts of speech
    tokens = nltk.word_tokenize(title)
    pos_tags = nltk.pos_tag(tokens)

    # Filter out verbs (VB, VBD, VBG, VBN, VBP, VBZ) and prepositions (IN)
    filtered_tokens = [word for word, pos in pos_tags if pos not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'IN']]

    # Reconstruct title
    title = ' '.join(filtered_tokens)

    # Correct spelling and unify 'colour'/'couleur' to 'color'
    corrected_title = ' '.join(spell_checker.correction(word) if word.lower() in ['colour', 'couleur'] else word for word in title.split())
    corrected_title = re.sub(r'\b(colour|couleur)\b', 'color', corrected_title, flags=re.IGNORECASE)

    # Apply title case to remaining words
    corrected_title = ' '.join(word.capitalize() for word in corrected_title.split() if word)

    return corrected_title

def normalize_text(text: str) -> str:
    """
    Clean text for fuzzy matching.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    return text

def extract_normalized_value(title: str, primary_attr: str, secondary_attr: str) -> str:
    """
    Extract the normalized value based on attributes.

    Args:
        title (str): The preprocessed title.
        primary_attr (str): The Primary Attribute.
        secondary_attr (str): The Secondary Attribute.

    Returns:
        str: The normalized value.
    """
    title_lower = title.lower()
    normalized = title

    if "dnu - 2xs-3xl" in title_lower or re.search(r'\b2xs-3xl\b', title_lower):
        normalized = "Size"
    elif "arne collection" in title_lower:
        normalized = "Collection"
    elif primary_attr == "Size":
        normalized = "Size"
        size_match = re.search(r'\b(xs|s|m|l|xl|xxl|2xs-3xl)\b', title_lower)
        if size_match:
            normalized = size_match.group(0).upper()
        elif re.search(r'\b(width|length|height|diameter|fan size|belt size)\b', title_lower):
            normalized = "Size"
    elif primary_attr == "Color":
        color_match = re.search(r'\b(black|white|red|blue|green|yellow|pink|grey|gray|olive|cream|mulberry|silver|charcoal|neon)\b', title_lower)
        if color_match:
            normalized = color_match.group(0).capitalize()
        elif "metal" in title_lower or "frame" in title_lower:
            normalized = "Color"
    elif primary_attr == "Material" and secondary_attr in ["Fill", "Cover", "Base"]:
        material_match = re.search(r'\b(cotton|goose down|leather|polyester|metal|flat bar)\b', title_lower)
        if material_match:
            normalized = material_match.group(0).capitalize()
    elif primary_attr == "Style" and secondary_attr == "Design":
        design_match = re.search(r'\b(paw|shoe|wine glass|charm|diy floor kit)\b', title_lower)
        if design_match:
            normalized = f"{design_match.group(0).capitalize()} Design"
    elif primary_attr in ["Letter", "Language"]:
        normalized = "Letter"
    elif primary_attr == "Finish" and "color" in title_lower:
        normalized = "Color & Finish"

    return normalized

def match_to_taxonomy(title: str, taxonomy: dict, regex_patterns: dict) -> tuple[str, str]:
    """
    Match the title to the taxonomy with precise attribute segmentation.

    Args:
        title (str): The preprocessed title.
        taxonomy (dict): The taxonomy dictionary.
        regex_patterns (dict): The regex patterns.

    Returns:
        tuple[str, str]: Primary Attribute and Secondary Attribute.
    """
    normalized_title = normalize_text(title)
    tokens = nltk.word_tokenize(normalized_title)
    
    primary_attr = ""
    secondary_attr = ""

    matched_attrs = []
    for attr, pattern in regex_patterns.items():
        if re.search(pattern, normalized_title, re.IGNORECASE):
            matched_attrs.append(attr)

    best_match = None
    best_score = 0
    def search_taxonomy(tax: dict | list, parent: str = ""):
        nonlocal best_match, best_score
        if isinstance(tax, list):
            for term in tax:
                score = fuzz.partial_ratio(normalized_title, term)
                if score > best_score and score >= 85:
                    best_score = score
                    best_match = parent
        elif isinstance(tax, dict):
            for key, value in tax.items():
                score = fuzz.partial_ratio(normalized_title, key.lower())
                if score > best_score and score >= 85:
                    best_score = score
                    best_match = key
                search_taxonomy(value, key)

    search_taxonomy(taxonomy)

    if best_match:
        primary_attr = best_match
        if best_match in ["Fill", "Cover", "Base"]:
            primary_attr = "Material"
            secondary_attr = best_match
        elif best_match in ["Units", "Pack Type"]:
            primary_attr = "Pack Size"
            secondary_attr = best_match

    if len(matched_attrs) > 1:
        primary_attr = matched_attrs[0]
        if "color" in title.lower() and "material" in matched_attrs:
            secondary_attr = "Color"
        elif "metal" in title.lower() and primary_attr in ["Size", "Width"]:
            secondary_attr = "Color"
        elif "finish" in title.lower() and "color" in matched_attrs:
            secondary_attr = "Finish"
        else:
            secondary_attr = matched_attrs[1] if len(matched_attrs) > 1 else ""
    elif len(matched_attrs) == 1 and not primary_attr:
        primary_attr = matched_attrs[0]

    if "metal" in normalized_title and primary_attr in ["Size", "Width"]:
        secondary_attr = "Color"
    if "color & finish" in normalized_title.lower():
        primary_attr = "Color"
        secondary_attr = "Finish"

    # Flag as ambiguous if no clear match
    if not primary_attr and not any(matched_attrs):
        return "Ambiguous", ""

    return primary_attr, secondary_attr

def process_title(row: pd.Series, row_id: int, taxonomy: dict, regex_patterns: dict) -> dict:
    """
    Process a single title.

    Args:
        row (pd.Series): The row of the DataFrame.
        row_id (int): The row ID.
        taxonomy (dict): The taxonomy dictionary.
        regex_patterns (dict): The regex patterns.

    Returns:
        dict: The processed title data.
    """
    title = str(row['variant_title']).strip()
    
    if not title:
        logging.warning(f"Row {row_id}: Empty title encountered")
        return {
            "Original Title": title,
            "Normalized Value": "",
            "Primary Attribute": "",
            "Secondary Attribute": "",
            "Is Ambiguous": True
        }

    preprocessed_title = preprocess_title(title)
    if not preprocessed_title:
        logging.warning(f"Row {row_id}: Preprocessed title is empty")
        return {
            "Original Title": title,
            "Normalized Value": "",
            "Primary Attribute": "",
            "Secondary Attribute": "",
            "Is Ambiguous": True
        }

    primary_attr, secondary_attr = match_to_taxonomy(preprocessed_title, taxonomy, regex_patterns)
    normalized_value = extract_normalized_value(preprocessed_title, primary_attr, secondary_attr)
    is_ambiguous = primary_attr == "Ambiguous"

    return {
        "Original Title": title,
        "Normalized Value": normalized_value,
        "Primary Attribute": primary_attr,
        "Secondary Attribute": secondary_attr,
        "Is Ambiguous": is_ambiguous
    }

def milestone2_simple_batch(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Process all variant titles for Milestone 2.

    Args:
        input_file (str): Path to the input Excel file.
        output_file (str): Path to the output Excel file.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Load and validate the full dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")

    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        if 'variant_title' not in df.columns:
            raise ValueError("Input file must contain a 'variant_title' column")
        logging.info(f"Loaded {len(df)} rows from {input_file}")
        if len(df) != 11522:
            logging.warning(f"Expected 11,522 rows, but found {len(df)} rows")
    except Exception as e:
        logging.error(f"Error loading file {input_file}: {e}")
        raise

    # Dynamic batch sizing with conservative memory usage
    available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
    batch_size = max(200, min(1000, int(available_memory / 20)))
    logging.info(f"Set batch size to {batch_size} based on {available_memory:.2f} MB available memory")

    results = []
    for start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[start:start + batch_size]
        batch_results = Parallel(n_jobs=min(2, os.cpu_count() or 1))(
            delayed(process_title)(row, row_id=start + idx + 1, taxonomy=TAXONOMY, regex_patterns=REGEX_PATTERNS)
            for idx, row in batch.iterrows()
        )
        results.extend(batch_results)
        logging.info(f"Processed batch {start // batch_size + 1}/{len(df) // batch_size + 1}")

    result_df = pd.DataFrame(results)

    # Validate output
    if len(result_df) != len(df):
        logging.warning(f"Output row count mismatch: Expected {len(df)}, got {len(result_df)}")

    # Save the full output
    try:
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Full batch output saved to {output_file} with {len(result_df)} rows")
    except Exception as e:
        logging.error(f"Error saving output to {output_file}: {e}")
        raise

    logging.info(f"Milestone 2 Completed: Processed {len(result_df)} titles.")

    return result_df

if __name__ == "__main__":
    """
    README:
    This script implements Milestone 2 of the Variant Title Normalization project.
    It processes all 11,522 titles from the input file and outputs the required fields.

    Requirements:
    - Install dependencies: pip install pandas fuzzywuzzy python-Levenshtein nltk openpyxl psutil tqdm joblib pyspellchecker
    - Input file:
      - 'variant_data.xlsx': Full dataset with 11,522 titles and a 'variant_title' column.
    - Output file:
      - 'milestone2_normalized_titles_simple.xlsx': Contains Original Title, Normalized Value, Primary Attribute, Secondary Attribute, and Is Ambiguous.

    To run:
    1. Ensure the input file is in the same directory as this script.
    2. Run the script:
       - python milestone2_variant_normalizer.py

    To fix console buffer issue in PowerShell:
    1. Increase the buffer size:
       - Right-click PowerShell title bar > Properties > Layout
       - Set 'Screen Buffer Size' Height to 9999
    2. Or redirect output to a file:
       - python milestone2_variant_normalizer.py > output.log
    3. Or use Command Prompt or VS Code terminal instead.
    """
    input_file = "variant_data.xlsx"
    output_file = "milestone2_normalized_titles_simple.xlsx"
    milestone2_simple_batch(input_file, output_file)