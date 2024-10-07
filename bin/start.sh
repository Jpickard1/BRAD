
echo "Starting Brad!!"
# PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS=True python3 -m BRAD.brad
npm start --prefix /usr/src/brad/brad-chat&
flask --app app run --host=0.0.0.0&