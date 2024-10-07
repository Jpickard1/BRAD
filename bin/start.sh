
echo "Starting Brad!!"
PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS=True python3 -m BRAD.brad
flask --app app run --host=0.0.0.0