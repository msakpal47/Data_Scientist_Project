import os
from flask import Flask


def create_app():
    base_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(base_dir, "templates")
    static_dir = os.path.join(base_dir, "static")
    app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    return app


