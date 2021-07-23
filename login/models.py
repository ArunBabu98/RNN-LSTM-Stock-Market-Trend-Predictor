from django.db import models

# Create your models here.

class login(models.Model):
    username = models.TextField()
    password = models.TextField()
    flag = models.BooleanField(default=False)

