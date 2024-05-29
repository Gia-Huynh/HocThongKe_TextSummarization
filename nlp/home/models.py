from django.db import models
from django.utils import timezone


# Create your models here.
class Works(models.Model):
    id_post = models.AutoField(primary_key=True)
    text = models.TextField(null=True, blank=True)
    text_summary = models.TextField(null=True, blank=True)
    data_save = models.DateTimeField(default=timezone.now)
