"""Sample module."""

import os

MAX = 10


# Adds numbers.
def add(a, b):
    return a + b


class Greeter:
    """Says hello."""

    greeting = "hi"

    def __init__(self, name):
        self.name = name

    @staticmethod
    def shout(text):
        return text.upper()


async def fetch():
    return await os.something()
