from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def augmented_test_prompts():
    base_prompts = _load_lines("500_prompts.txt")
    base_prompt = random.choice(base_prompts)
    style = "easily transformed into an SVG . looks like an clipart or icon . limited lineal colors . cartoony . simple solid shapes and flat colors . simple singular subject . flat background . objects represented through simple shapes"
    # Return both the full styled prompt and metadata with the base prompt
    return f"depicting a {base_prompt} {style}", {"base_prompt": base_prompt}

def styled_animals():
    animals = _load_lines("simple_animals.txt")
    animal = random.choice(animals)
    # style = "lineal color space . cartoony . simple shapes. limited colors . solid shapes and flat colors . simple scene . flat background . objects represented through simple shapes . looks like a colored in a toddler coloring book"
    # style = "lineal color space . childrens cartoon . simple shapes. limited colors . solid shapes and flat colors . simple scene . flat background . objects represented through simple shapes"
    style = "easily transformed into an SVG . looks like an clipart or icon . limited lineal colors . cartoony . simple solid shapes and flat colors . simple singular subject . flat background . objects represented through simple shapes"
    # Return both the full styled prompt and metadata with the animal name
    return f"depicting a {animal} {style}", {"base_animal": animal}

def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
