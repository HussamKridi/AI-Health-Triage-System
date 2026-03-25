from flask import Flask
from routes import routes_bp
from api import api_bp
from models import init_db

app = Flask(__name__)
app.secret_key = 'super_secret_health_key'

init_db()
app.register_blueprint(routes_bp)
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True, port=5000)