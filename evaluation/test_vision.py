"""Quick test script for the vision analysis module.

Usage:
    python evaluation/test_vision.py <image_path> [description]

Examples:
    python evaluation/test_vision.py test_plant.jpg
    python evaluation/test_vision.py test_plant.jpg "leaves are turning yellow"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent.modules.vision import analyze_image


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python evaluation/test_vision.py <image_path> [description]")
        print("\nProvide a path to a plant image to analyze.")
        sys.exit(1)

    image_path = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else ""

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing: {image_path}")
    if description:
        print(f"User note: {description}")
    print("-" * 50)

    result = analyze_image(image_path, description)

    print(f"\nPlant type:  {result.plant_type_guess}")
    print(f"Confidence:  {result.confidence:.0%}")
    print(f"Symptoms:    {', '.join(result.symptoms) or 'None detected'}")
    print(f"\nDescription:\n{result.raw_description}")
    print("-" * 50)
    print("\nFull JSON output:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
