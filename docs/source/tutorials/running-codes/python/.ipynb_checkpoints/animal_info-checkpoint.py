"""
This script provides information about different animals based on user input.

Usage:
    python animal_info.py <animal_name> <age> <weight> [<color>]

Arguments:
    <animal_name>  : Name of the animal (e.g., "Lion", "Elephant")
    <age>          : Age of the animal in years (e.g., 5)
    <weight>       : Weight of the animal in kilograms (e.g., 200)
    [<color>]      : Optional color of the animal (e.g., "Golden")
"""

import sys

def get_animal_info(animal_name, age, weight, color=None):
    """
    Print information about the animal.

    Args:
        animal_name (str): Name of the animal
        age (int): Age of the animal in years
        weight (float): Weight of the animal in kilograms
        color (str, optional): Color of the animal
    """
    info = f"Animal: {animal_name}\nAge: {age} years\nWeight: {weight} kg"
    if color:
        info += f"\nColor: {color}"
    print(info)

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python animal_info.py <animal_name> <age> <weight> [<color>]")
        sys.exit(1)
    
    animal_name = sys.argv[1]
    age = int(sys.argv[2])
    weight = float(sys.argv[3])
    color = sys.argv[4] if len(sys.argv) == 5 else None
    
    get_animal_info(animal_name, age, weight, color)

if __name__ == "__main__":
    main()
