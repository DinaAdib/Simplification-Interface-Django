
from django.conf.urls import include, url
from django.contrib import admin

from .views import *

urlpatterns = [
    url('', view=index),
    url('/summarize', view=summarize),
]