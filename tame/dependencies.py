from fastapi import Request

from app import TAMEApplication


def get_tame_app(request: Request) -> TAMEApplication:
    return request.app.state.tame
