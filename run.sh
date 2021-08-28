if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	export FLASK_APP=app.py
	export FLASK_ENV='development'
else
	set FLASK_APP=app.py
	set FLASK_ENV='development'
fi

flask run