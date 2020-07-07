from flask import Flask, render_template, request, redirect, url_for, jsonify
from googletrans import Translator


translator = Translator()


def translate(input_sent):

    input_sent_og = input_sent
    decoded_sent_hi = translator.translate(input_sent_og, dest='hi').text

    return decoded_sent_hi