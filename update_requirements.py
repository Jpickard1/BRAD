import toml

# Load pyproject.toml
with open('pyproject.toml', 'r') as file:
    pyproject = toml.load(file)

# Load requirements.txt
with open('requirements.txt', 'r') as file:
    requirements = file.read().splitlines()

# Update dependencies in pyproject.toml
pyproject['project']['dependencies'] = requirements

# Save the updated pyproject.toml
with open('pyproject.toml', 'w') as file:
    toml.dump(pyproject, file)
